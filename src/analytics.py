"""
analytics.py
Temporal analysis and driver-state classification.

DrowsinessAnalytics receives one (EAR, MAR) pair per frame and keeps a
short history so it can tell a normal blink from real drowsiness.

Signals it tracks:
    * closed-eye streak : consecutive frames with EAR below the threshold.
    * yawn streak       : consecutive frames with MAR above the threshold.
    * PERCLOS           : fraction of frames in a rolling window where the
                          eyes were closed.

State machine:
    CRITICAL_DROWSY : eyes closed for the full streak length, or PERCLOS
                      at or above the critical cutoff.
    MILD_FATIGUE    : PERCLOS at or above the mild cutoff, or a sustained yawn.
    ALERT           : none of the above.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from src import config


@dataclass(frozen=True)
class AnalyticsResult:
    """Snapshot of the analytics after processing one frame."""

    state: str               # one of config.STATE_* values
    perclos: float           # fraction between 0.0 and 1.0
    eyes_closed: bool        # eyes below threshold in this frame
    yawning: bool            # mouth above threshold in this frame
    closed_streak: int       # consecutive closed-eye frames so far
    yawn_count: int          # total yawns detected this session
    face_detected: bool      # False when the frame had no face


class DrowsinessAnalytics:
    """Rolling-window drowsiness analytics with a three-state classifier."""

    def __init__(
        self,
        ear_threshold: float = config.EAR_THRESHOLD,
        mar_threshold: float = config.MAR_THRESHOLD,
        consec_frames: int = config.CONSEC_FRAMES_THRESHOLD,
        window_size: int = config.PERCLOS_WINDOW_SIZE,
        perclos_mild: float = config.PERCLOS_MILD_THRESHOLD,
        perclos_critical: float = config.PERCLOS_CRITICAL_THRESHOLD,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.consec_frames = consec_frames
        self.window_size = window_size
        self.perclos_mild = perclos_mild
        self.perclos_critical = perclos_critical

        # PERCLOS is reported as 0 until the window holds this many frames.
        # Without this, the very first closed frame would read as 100%.
        self.min_samples = max(1, window_size // 10)

        # One boolean per frame: True means the eyes were closed.
        self._eye_window: Deque[bool] = deque(maxlen=window_size)

        self._closed_streak = 0
        self._yawn_streak = 0
        self._missing_frames = 0
        self.yawn_count = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def update(self, ear: Optional[float], mar: Optional[float]) -> AnalyticsResult:
        """
        Feed one frame's measurements and get the current driver state.

        Args:
            ear: eye aspect ratio, or None if no face was found.
            mar: mouth aspect ratio, or None if no face was found.
        """
        if ear is None or mar is None:
            return self._handle_missing_face()

        self._missing_frames = 0

        eyes_closed = ear < self.ear_threshold
        yawning = mar > self.mar_threshold

        # Eye tracking
        self._eye_window.append(eyes_closed)
        self._closed_streak = self._closed_streak + 1 if eyes_closed else 0

        # Yawn tracking: count each yawn once, when its streak first
        # reaches the threshold, so a long yawn is not counted repeatedly.
        if yawning:
            self._yawn_streak += 1
            if self._yawn_streak == self.consec_frames:
                self.yawn_count += 1
        else:
            self._yawn_streak = 0

        return self._build_result(eyes_closed, yawning, face_detected=True)

    def compute_perclos(self) -> float:
        """Return the fraction of recent frames with closed eyes (0.0 to 1.0)."""
        if len(self._eye_window) < self.min_samples:
            return 0.0
        return sum(self._eye_window) / len(self._eye_window)

    def reset(self) -> None:
        """Clear all history, for example when the driver changes."""
        self._eye_window.clear()
        self._closed_streak = 0
        self._yawn_streak = 0
        self._missing_frames = 0
        self.yawn_count = 0

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    def _classify(self, perclos: float) -> str:
        """State machine: the most severe matching condition wins."""
        if (
            self._closed_streak >= self.consec_frames
            or perclos >= self.perclos_critical
        ):
            return config.STATE_CRITICAL_DROWSY

        if perclos >= self.perclos_mild or self._yawn_streak >= self.consec_frames:
            return config.STATE_MILD_FATIGUE

        return config.STATE_ALERT

    def _build_result(
        self, eyes_closed: bool, yawning: bool, face_detected: bool
    ) -> AnalyticsResult:
        perclos = self.compute_perclos()
        return AnalyticsResult(
            state=self._classify(perclos),
            perclos=perclos,
            eyes_closed=eyes_closed,
            yawning=yawning,
            closed_streak=self._closed_streak,
            yawn_count=self.yawn_count,
            face_detected=face_detected,
        )

    def _handle_missing_face(self) -> AnalyticsResult:
        """
        Handle a frame with no face.

        Missing frames are not added to the PERCLOS window, so leaving the
        frame is never mistaken for closed eyes. Streaks reset because they
        require consecutive frames. If the face stays missing for a whole
        window length, the old history is stale and gets cleared.
        """
        self._missing_frames += 1
        self._closed_streak = 0
        self._yawn_streak = 0

        if self._missing_frames >= self.window_size:
            self._eye_window.clear()

        return self._build_result(
            eyes_closed=False, yawning=False, face_detected=False
        )