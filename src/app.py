"""Minimal Streamlit webcam dashboard for manual frame sampling."""

import time

import streamlit as st

from src.analytics import DrowsinessAnalytics
from src.config import (
    CAMERA_INDEX,
    EAR_CLOSED_THRESHOLD,
    FACE_MESH_MAX_FACES,
    PERCLOS_ALERT_THRESHOLD,
    PERCLOS_WINDOW_SECONDS,
)
from src.detector import extract_face_metrics


def main() -> None:
    st.set_page_config(page_title="Driver Drowsiness Monitor", page_icon=":eye:")
    st.title("Driver Drowsiness Monitor")
    st.caption("Experimental monitoring prototype. Not a certified safety system.")

    if "analytics" not in st.session_state:
        st.session_state.analytics = DrowsinessAnalytics(
            window_seconds=PERCLOS_WINDOW_SECONDS,
            closed_eye_threshold=EAR_CLOSED_THRESHOLD,
            perclos_alert_threshold=PERCLOS_ALERT_THRESHOLD,
        )

    enable_camera = st.checkbox("Enable webcam capture")
    if not enable_camera:
        st.info("Enable the webcam to sample a frame.")
        return

    if not st.button("Capture frame"):
        return

    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        st.error(f"Camera dependencies are unavailable: {error}")
        return

    camera = cv2.VideoCapture(CAMERA_INDEX)
    try:
        success, frame = camera.read()
    finally:
        camera.release()
    if not success:
        st.error("Could not read a frame from the webcam. Check camera permissions and index.")
        return

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=FACE_MESH_MAX_FACES,
            refine_landmarks=True,
        ) as face_mesh:
            measurements = extract_face_metrics(frame, face_mesh)
    except Exception as error:
        st.error(f"Face landmark detection failed: {error}")
        return

    if measurements is None:
        st.warning("No face landmarks detected in this frame.")
        return

    ear, mar = measurements
    alert = st.session_state.analytics.update(time.monotonic(), ear)
    perclos = st.session_state.analytics.perclos()
    st.metric("Eye aspect ratio", f"{ear:.3f}")
    st.metric("Mouth aspect ratio", f"{mar:.3f}")
    st.metric("PERCLOS", f"{perclos:.1%}")
    if alert:
        st.error("Drowsiness threshold reached. Stop driving when safe and rest.")
    else:
        st.success("No alert threshold reached in the sampled observations.")


if __name__ == "__main__":
    main()