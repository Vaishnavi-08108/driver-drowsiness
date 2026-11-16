"""
detector.py
Facial feature extraction for the Driver Drowsiness Detection System.

FaceFeatureExtractor wraps MediaPipe Face Mesh and turns a raw webcam frame
into two numbers per frame:

    EAR (Eye Aspect Ratio)   : drops toward 0 when the eyes close.
    MAR (Mouth Aspect Ratio) : rises when the mouth opens wide (yawning).

It also draws the eye and mouth landmarks onto the frame for the dashboard.
"""

from typing import Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src import config


class FaceFeatureExtractor:
    """Extracts EAR and MAR from video frames using MediaPipe Face Mesh."""

    # -----------------------------------------------------------------------
    # Landmark index arrays (MediaPipe Face Mesh, 468 landmarks)
    # -----------------------------------------------------------------------
    # Eye points are ordered p1..p6 around the eye:
    #   p1 = outer corner, p2/p3 = upper lid, p4 = inner corner,
    #   p5/p6 = lower lid.
    # "Left" and "right" refer to the subject's own left and right eye.
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    # Outer lip contour, ordered p1..p8 going around the mouth:
    #   p1 = left corner, p2/p3/p4 = upper lip, p5 = right corner,
    #   p6/p7/p8 = lower lip.
    MOUTH = [61, 39, 0, 269, 291, 405, 17, 181]

    # BGR colors used for the overlay
    COLOR_EYE = (0, 255, 0)
    COLOR_MOUTH = (0, 165, 255)
    COLOR_MESH = (200, 200, 200)

    def __init__(self, draw_full_mesh: bool = False) -> None:
        """
        Args:
            draw_full_mesh: if True, draw the whole face tessellation in
                addition to the eye and mouth contours. Looks impressive
                but costs some frame rate.
        """
        self.draw_full_mesh = draw_full_mesh

        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_drawing = mp.solutions.drawing_utils
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,  # video mode: track faces between frames
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=config.REFINE_LANDMARKS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    # -----------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------
    @staticmethod
    def compute_aspect_ratio(points: Sequence[Sequence[float]]) -> float:
        """
        Compute an aspect ratio from an even-length ring of 2D points.

        The first point and the point halfway around the ring are the
        horizontal extremes (the corners). Every remaining point pairs with
        its mirror across the ring to form a vertical opening:

            6 points (eye)   : EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
            8 points (mouth) : MAR = (|p2-p8| + |p3-p7| + |p4-p6|) / (3 * |p1-p5|)

        Both formulas are the mean vertical opening divided by the
        horizontal width, so one method serves EAR and MAR.

        Args:
            points: N x 2 array-like of (x, y) pixel coordinates, N even, N >= 4.

        Returns:
            The aspect ratio as a float. Returns 0.0 if the width is zero.
        """
        pts = np.asarray(points, dtype=np.float64)
        n = len(pts)
        if n < 4 or n % 2 != 0:
            raise ValueError("points must contain an even number of points (>= 4)")

        half = n // 2
        horizontal = np.linalg.norm(pts[0] - pts[half])
        if horizontal == 0:
            return 0.0

        # Pair point i with point n - i for i = 1 .. half - 1
        vertical = [np.linalg.norm(pts[i] - pts[n - i]) for i in range(1, half)]
        return float(np.mean(vertical) / horizontal)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _landmarks_to_pixels(
        landmarks, indices: Sequence[int], width: int, height: int
    ) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates (N x 2 array).

        Using pixel units (not normalized 0..1 values) keeps distances
        correct on non-square frames such as 640x480.
        """
        return np.array(
            [(landmarks[i].x * width, landmarks[i].y * height) for i in indices],
            dtype=np.float64,
        )

    def _draw_contour(
        self, frame: np.ndarray, points: np.ndarray, color: Tuple[int, int, int]
    ) -> None:
        """Draw a closed outline with a dot on every landmark."""
        pts = points.astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
        """
        Detect the face in a BGR frame and compute EAR and MAR.

        Args:
            frame: BGR image from OpenCV (height x width x 3).

        Returns:
            (annotated_frame, ear, mar)
            ear and mar are floats, or None when no face is found. The
            caller should treat None as "no data" rather than as zero, so a
            driver who leaves the frame is not counted as having closed eyes.
        """
        height, width = frame.shape[:2]

        # MediaPipe expects RGB. Marking the array read-only lets it skip a copy.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame, None, None

        landmarks = results.multi_face_landmarks[0].landmark

        # Pixel coordinates for each feature
        left_eye = self._landmarks_to_pixels(landmarks, self.LEFT_EYE, width, height)
        right_eye = self._landmarks_to_pixels(landmarks, self.RIGHT_EYE, width, height)
        mouth = self._landmarks_to_pixels(landmarks, self.MOUTH, width, height)

        # Average both eyes so a single noisy eye does not dominate
        left_ear = self.compute_aspect_ratio(left_eye)
        right_ear = self.compute_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = self.compute_aspect_ratio(mouth)

        # Overlay
        if self.draw_full_mesh:
            self._mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=self._mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._mp_drawing.DrawingSpec(
                    color=self.COLOR_MESH, thickness=1
                ),
            )
        self._draw_contour(frame, left_eye, self.COLOR_EYE)
        self._draw_contour(frame, right_eye, self.COLOR_EYE)
        self._draw_contour(frame, mouth, self.COLOR_MOUTH)

        return frame, ear, mar

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    def close(self) -> None:
        """Release MediaPipe resources."""
        self._face_mesh.close()

    def __enter__(self) -> "FaceFeatureExtractor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()