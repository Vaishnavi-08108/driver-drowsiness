"""
app.py
Streamlit dashboard for the Driver Drowsiness Detection System.

Run from the project root:
    streamlit run src/app.py

Features:
    * Live webcam feed with eye and mouth landmark overlays.
    * Metric cards for EAR, MAR, PERCLOS, yawns, and alert counts.
    * Status badge: green (alert), yellow (mild fatigue), red (critical).
    * Sound alert in the browser while the driver is critically drowsy.
    * Auto-calibration of the EAR and MAR thresholds for the current driver.
    * Live chart of EAR and PERCLOS.
    * CSV event log per session, with a summary and a download button.
"""

import platform
import sys
import time
from collections import deque
from pathlib import Path

# Make "from src import ..." work when launched with `streamlit run src/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.alerts import SoundAlert
from src.analytics import AnalyticsResult, DrowsinessAnalytics
from src.calibration import CalibrationResult, Calibrator
from src.detector import FaceFeatureExtractor
from src.session_log import SessionLogger

# Refresh the metric cards and badge every N frames. Updating Streamlit
# widgets is slow compared with video processing, so this keeps FPS high.
UI_UPDATE_EVERY = 3

NO_FACE_COLOR = "#7f8c8d"       # grey badge when no face is visible
CALIBRATION_COLOR = "#3498db"   # blue badge while calibrating

STATE_LABELS = {
    config.STATE_ALERT: "ALERT",
    config.STATE_MILD_FATIGUE: "WARNING: MILD FATIGUE",
    config.STATE_CRITICAL_DROWSY: "DANGER: CRITICAL DROWSINESS",
}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def hex_to_bgr(hex_color: str) -> tuple:
    """Convert '#rrggbb' to an OpenCV (b, g, r) tuple."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .badge {
            padding: 14px 18px; border-radius: 12px; text-align: center;
            font-size: 1.5rem; font-weight: 700; color: #ffffff;
            letter-spacing: 0.5px;
        }
        .card {
            border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;
            background: rgba(128, 128, 128, 0.12);
            border-left: 6px solid var(--accent);
        }
        .card .label { font-size: 0.8rem; opacity: 0.7; text-transform: uppercase; }
        .card .value { font-size: 2rem; font-weight: 700; line-height: 1.2; }
        .card .sub   { font-size: 0.75rem; opacity: 0.6; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(label: str, color: str) -> str:
    return f'<div class="badge" style="background:{color};">{label}</div>'


def card_html(label: str, value: str, sub: str, accent: str) -> str:
    return (
        f'<div class="card" style="--accent:{accent};">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def perclos_color(perclos: float, mild: float, critical: float) -> str:
    if perclos >= critical:
        return config.STATE_COLORS[config.STATE_CRITICAL_DROWSY]
    if perclos >= mild:
        return config.STATE_COLORS[config.STATE_MILD_FATIGUE]
    return config.STATE_COLORS[config.STATE_ALERT]


# ---------------------------------------------------------------------------
# Session state and session lifecycle
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Create defaults once and apply thresholds from a finished calibration.

    This must run before the sidebar sliders are created, because Streamlit
    only lets you set a widget's value before the widget exists.
    """
    defaults = {
        "ear_threshold": config.EAR_THRESHOLD,
        "mar_threshold": config.MAR_THRESHOLD,
        "consec_frames": config.CONSEC_FRAMES_THRESHOLD,
        "logger": None,
        "last_summary": None,
        "last_csv": None,
        "last_csv_name": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if "chart_buf" not in st.session_state:
        st.session_state.chart_buf = deque(maxlen=config.CHART_HISTORY_FRAMES)

    pending = st.session_state.pop("pending_thresholds", None)
    if pending is not None:
        ear_t, mar_t = pending
        st.session_state["ear_threshold"] = round(ear_t, 2)
        st.session_state["mar_threshold"] = round(mar_t, 2)


def start_session() -> None:
    """Open a new CSV log and clear the previous session's leftovers."""
    st.session_state.logger = SessionLogger()
    st.session_state.last_summary = None
    st.session_state.last_csv = None
    st.session_state.last_csv_name = None
    st.session_state.chart_buf.clear()


def close_session() -> None:
    """Finish the CSV log and keep its summary for the stopped screen."""
    logger = st.session_state.logger
    if logger is None:
        return
    logger.close()
    st.session_state.last_summary = logger.summary_text()
    st.session_state.last_csv = logger.read_bytes()
    st.session_state.last_csv_name = logger.path.name
    st.session_state.logger = None


def finish_calibration(
    result: CalibrationResult, logger: SessionLogger, ui: dict
) -> None:
    """Store the calibration outcome and rerun so the sliders update."""
    logger.log_calibration(result)
    st.session_state["calib_message"] = (result.success, result.message)
    if result.success:
        st.session_state["pending_thresholds"] = (
            result.ear_threshold, result.mar_threshold,
        )
    ui["calib"].empty()
    st.rerun()  # stops this run; the next run starts with the new sliders


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_ui(
    ui: dict,
    frame,
    result: AnalyticsResult,
    ear,
    mar,
    label: str,
    color: str,
    fps: float,
    settings: dict,
    logger: SessionLogger,
) -> None:
    """Push the latest frame, badge, and metric cards to the page."""
    # No width argument on purpose: the width options for st.image differ
    # between Streamlit versions, and the frame is already 640x480.
    ui["video"].image(frame, channels="BGR")
    ui["badge"].markdown(badge_html(label, color), unsafe_allow_html=True)

    ear_text = "--" if ear is None else f"{ear:.2f}"
    mar_text = "--" if mar is None else f"{mar:.2f}"
    ear_accent = (
        config.STATE_COLORS[config.STATE_CRITICAL_DROWSY]
        if result.eyes_closed else config.STATE_COLORS[config.STATE_ALERT]
    )
    mar_accent = (
        config.STATE_COLORS[config.STATE_MILD_FATIGUE]
        if result.yawning else config.STATE_COLORS[config.STATE_ALERT]
    )

    ui["ear"].markdown(
        card_html("Eye aspect ratio", ear_text,
                  f"closed below {settings['ear_threshold']:.2f}", ear_accent),
        unsafe_allow_html=True,
    )
    ui["mar"].markdown(
        card_html("Mouth aspect ratio", mar_text,
                  f"yawn above {settings['mar_threshold']:.2f}", mar_accent),
        unsafe_allow_html=True,
    )
    ui["perclos"].markdown(
        card_html(
            "PERCLOS", f"{result.perclos * 100:.1f}%",
            f"window: {config.PERCLOS_WINDOW_SIZE} frames",
            perclos_color(result.perclos, config.PERCLOS_MILD_THRESHOLD,
                          config.PERCLOS_CRITICAL_THRESHOLD),
        ),
        unsafe_allow_html=True,
    )
    ui["yawns"].markdown(
        card_html(
            "Yawns detected", str(logger.yawn_total),
            f"warnings: {logger.warning_count}, "
            f"critical alerts: {logger.critical_count}",
            config.STATE_COLORS[config.STATE_ALERT],
        ),
        unsafe_allow_html=True,
    )
    ui["info"].caption(f"{fps:.1f} FPS")


def render_chart(slot, buffer: deque) -> None:
    """Draw the recent EAR, PERCLOS, and EAR threshold as a line chart."""
    if not buffer:
        return
    data = pd.DataFrame(list(buffer), columns=["EAR", "PERCLOS", "EAR threshold"])
    slot.line_chart(data, height=220)

# ---------------------------------------------------------------------------
# Camera and detector that survive Streamlit reruns
# ---------------------------------------------------------------------------
def get_camera(index: int):
    """Return an open camera, reusing the one from the previous rerun."""
    cam = st.session_state.get("camera")
    if (
        cam is not None
        and st.session_state.get("camera_open_index") == index
        and cam.isOpened()
    ):
        return cam

    release_camera()
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cam = cv2.VideoCapture(index, backend)
    if not cam.isOpened():
        cam.release()
        return None
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    st.session_state["camera"] = cam
    st.session_state["camera_open_index"] = index
    return cam


def release_camera() -> None:
    cam = st.session_state.pop("camera", None)
    if cam is not None:
        cam.release()
    st.session_state.pop("camera_open_index", None)


def get_extractor() -> FaceFeatureExtractor:
    """Return one FaceFeatureExtractor that is created once and reused."""
    extractor = st.session_state.get("extractor")
    if extractor is None:
        extractor = FaceFeatureExtractor()
        st.session_state["extractor"] = extractor
    return extractor


def release_extractor() -> None:
    extractor = st.session_state.pop("extractor", None)
    if extractor is not None:
        extractor.close()
# ---------------------------------------------------------------------------
# Monitoring loop
# ---------------------------------------------------------------------------
def run_monitor(settings: dict, logger: SessionLogger, ui: dict) -> None:
    """Capture frames, run detection and analytics, and update the UI."""
    cap = get_camera(settings["camera_index"])
    if cap is None:
        st.error(
            f"Could not open camera {settings['camera_index']}. "
            "Close other apps that use the webcam, or try another index."
        )
        return

    analytics = DrowsinessAnalytics(
        ear_threshold=settings["ear_threshold"],
        mar_threshold=settings["mar_threshold"],
        consec_frames=settings["consec_frames"],
    )
    extractor = get_extractor()
    extractor.draw_full_mesh = settings["full_mesh"]
    calibrator = Calibrator()
    sound = SoundAlert(enabled=settings["sound"])
    chart_buffer = st.session_state.chart_buf

    if settings["calibrate"]:
        calibrator.start()

    frame_count = 0
    fps = 0.0
    last_tick = time.time()

    # No try/finally here on purpose. The camera and extractor stay open
    # when Streamlit reruns the script, and main() releases them when
    # monitoring is switched off.
    while True:
        ok, frame = cap.read()
        if not ok:
            release_camera()
            st.error("Lost the camera feed.")
            break

        frame = cv2.flip(frame, 1)  # mirror view feels natural
        frame, ear, mar = extractor.process_frame(frame)
        frame_count += 1
        refresh = frame_count % UI_UPDATE_EVERY == 0

        # ----- Calibration mode: collect samples, skip alerts -----
        if calibrator.active:
            if calibrator.add_sample(ear, mar):
                finish_calibration(calibrator.result, logger, ui)
            cv2.putText(
                frame, "CALIBRATING: eyes open, mouth closed", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                hex_to_bgr(CALIBRATION_COLOR), 2, cv2.LINE_AA,
            )
            if refresh:
                remaining = config.CALIBRATION_DURATION_SEC * (
                    1.0 - calibrator.progress()
                )
                ui["video"].image(frame, channels="BGR")
                ui["badge"].markdown(
                    badge_html("CALIBRATING", CALIBRATION_COLOR),
                    unsafe_allow_html=True,
                )
                ui["calib"].progress(
                    calibrator.progress(),
                    text=f"Look at the camera and hold still: "
                         f"{remaining:.0f}s left",
                )
            continue

        # ----- Normal monitoring -----
        result: AnalyticsResult = analytics.update(ear, mar)
        logger.update(result, ear, mar)
        sound.update(result.state, ui["sound"])
        chart_buffer.append((
            np.nan if ear is None else ear,
            result.perclos,
            settings["ear_threshold"],
        ))

        # Status text burned into the video frame
        if result.face_detected:
            label = STATE_LABELS[result.state]
            color = config.STATE_COLORS[result.state]
        else:
            label, color = "NO FACE DETECTED", NO_FACE_COLOR
        cv2.putText(
            frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, hex_to_bgr(color), 2, cv2.LINE_AA,
        )

        if refresh:
            now = time.time()
            fps = UI_UPDATE_EVERY / max(now - last_tick, 1e-6)
            last_tick = now
            render_ui(ui, frame, result, ear, mar, label, color, fps,
                      settings, logger)

        if frame_count % config.CHART_UPDATE_EVERY == 0:
            render_chart(ui["chart"], chart_buffer)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Driver Drowsiness Monitor",
        page_icon="🚗",
        layout="wide",
    )
    inject_css()
    st.title("Driver Drowsiness Detection System")

    init_session_state()
    calib_message = st.session_state.pop("calib_message", None)

    # ----- Sidebar -----
    with st.sidebar:
        st.header("Controls")
        running = st.toggle("Start monitoring", value=False)

        st.subheader("Camera")
        camera_index = st.number_input(
            "Camera index", min_value=0, max_value=10,
            value=config.CAMERA_INDEX, step=1,
        )
        full_mesh = st.checkbox("Draw full face mesh", value=False)

        st.subheader("Thresholds")
        # No `value` argument: the starting values come from session state,
        # which is how calibration can move the sliders.
        ear_threshold = st.slider(
            "EAR threshold", min_value=config.EAR_THRESHOLD_MIN,
            max_value=config.EAR_THRESHOLD_MAX, step=0.01,
            key="ear_threshold",
        )
        mar_threshold = st.slider(
            "MAR threshold", min_value=config.MAR_THRESHOLD_MIN,
            max_value=config.MAR_THRESHOLD_MAX, step=0.01,
            key="mar_threshold",
        )
        consec_frames = st.slider(
            "Consecutive frames", min_value=5, max_value=60, step=1,
            key="consec_frames",
        )

        st.subheader("Calibration")
        st.caption(
            f"Look at the camera with eyes open and mouth closed for "
            f"{config.CALIBRATION_DURATION_SEC} seconds."
        )
        calibrate_clicked = st.button("Calibrate my thresholds")
        if calibrate_clicked and not running:
            st.warning("Turn on Start monitoring first.")

        st.subheader("Alerts")
        sound_on = st.checkbox("Sound alert on critical drowsiness", value=True)

    settings = {
        "camera_index": int(camera_index),
        "full_mesh": full_mesh,
        "ear_threshold": ear_threshold,
        "mar_threshold": mar_threshold,
        "consec_frames": consec_frames,
        "sound": sound_on,
        "calibrate": bool(calibrate_clicked and running),
    }

    # ----- Session lifecycle (one CSV log per monitoring session) -----
    if running and st.session_state.logger is None:
        start_session()
    elif not running and st.session_state.logger is not None:
        close_session()

    if calib_message is not None:
        calib_ok, calib_text = calib_message
        if calib_ok:
            st.success(calib_text)
        else:
            st.warning(calib_text)

    # ----- Layout -----
    badge_ph = st.empty()
    calib_ph = st.empty()
    left, right = st.columns([3, 2])
    with left:
        video_ph = st.empty()
        info_ph = st.empty()
    with right:
        row1 = st.columns(2)
        row2 = st.columns(2)
        ear_ph = row1[0].empty()
        mar_ph = row1[1].empty()
        perclos_ph = row2[0].empty()
        yawns_ph = row2[1].empty()

    st.subheader("Live EAR and PERCLOS")
    chart_ph = st.empty()
    sound_ph = st.empty()   # holds the invisible <audio> element for beeps

    ui = {
        "badge": badge_ph, "calib": calib_ph, "video": video_ph,
        "info": info_ph, "ear": ear_ph, "mar": mar_ph,
        "perclos": perclos_ph, "yawns": yawns_ph,
        "chart": chart_ph, "sound": sound_ph,
    }

    if running:
        run_monitor(settings, st.session_state.logger, ui)
    else:
        release_camera()
        release_extractor()
        badge_ph.markdown(badge_html("MONITORING STOPPED", NO_FACE_COLOR),
                          unsafe_allow_html=True)
        video_ph.info("Turn on **Start monitoring** in the sidebar to begin.")

        # ...the "Last session summary" block stays exactly as it was
        if st.session_state.last_summary:
            st.subheader("Last session summary")
            st.text(st.session_state.last_summary)
            st.download_button(
                "Download CSV log",
                data=st.session_state.last_csv,
                file_name=st.session_state.last_csv_name,
                mime="text/csv",
            )


main()