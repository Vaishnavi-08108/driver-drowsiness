"""
app.py
Unified Streamlit Dashboard for the Driver Drowsiness Detection System.

Tabs:
    1. Live Monitor & Driver Safety: Real-time webcam monitoring, facial landmark tracking,
       EAR/MAR/PERCLOS cards, auto-calibration, emergency countdown banner, and "I'm Awake" button.
    2. Travel Readiness & Sleep Advisor: Sleep score breakdown (0-100), sleep deficit & debt tracking,
       interactive sleep logging form, and past sleep history.
    3. Emergency Contacts & Geolocation: Telegram emergency contacts (CRUD up to 5 contacts), masked
       chat IDs, Telegram test alert trigger, and GPS/IP coordinates with Google Maps links.
    4. Driver Fatigue AI Assistant: Interactive conversational assistant with preset questions,
       Anthropic Claude integration, and offline sleep science expert knowledge.
"""

import datetime as dt
import json
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

from src import config
from src.alerts import SoundAlert, generate_beep_wav
from src.analytics import AnalyticsResult, DrowsinessAnalytics
from src.calibration import CalibrationResult, Calibrator
from src.chatbot import FatigueChatBot, SUGGESTED_PROMPTS
from src.contacts import ContactBook, mask_chat_id, MAX_CONTACTS
from src.detector import FaceFeatureExtractor
from src.emergency import (
    EVENT_ALERT_DONE,
    EVENT_ALERT_SENDING,
    EVENT_COUNTDOWN_RECOVERED,
    EVENT_COUNTDOWN_STARTED,
    PHASE_COOLDOWN,
    PHASE_COUNTDOWN,
    PHASE_IDLE,
    PHASE_SENDING,
    EmergencyManager,
    TelegramClient,
)
from src.location import (
    BROWSER_MAX_AGE_SEC,
    SOURCE_BROWSER,
    SOURCE_IP,
    SOURCE_MANUAL,
    LocationProvider,
    describe,
    map_link,
)
from src.session_log import SessionLogger
from src.sleep_advisor import SleepAdvisor, SleepLogBook, ReadinessScorer

UI_UPDATE_EVERY = 3
NO_FACE_COLOR = "#64748b"
BRAND = """<div class="driveguard-brand">
<svg width="30" height="30" viewBox="0 0 32 32" fill="none" stroke="currentColor"
     stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
  <circle cx="16" cy="16" r="12"/><circle cx="16" cy="16" r="3"/>
  <path d="M5 13l8 2m6 0 8-2M16 19v9"/>
</svg><span>DriveGuard</span></div>"""
PAGES = ("Monitoring", "Emergency", "Sleep Assistant", "Travel Readiness")
CALIBRATION_COLOR = "#2563eb"

STATE_LABELS = {
    config.STATE_ALERT: "ALERT",
    config.STATE_MILD_FATIGUE: "WARNING: MILD FATIGUE",
    config.STATE_CRITICAL_DROWSY: "DANGER: CRITICAL DROWSINESS",
}


def hex_to_bgr(hex_color: str) -> tuple:
    """Convert '#rrggbb' to an OpenCV (b, g, r) tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (4, 2, 0))


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

        :root {
            color-scheme: light;
            --clinical-bg: #FAFBF9;
            --clinical-surface: #FFFFFF;
            --clinical-text: #18312E;
            --clinical-muted: #718783;
            --clinical-border: #DDE7E4;
            --clinical-primary: #17583F;
            --clinical-primary-soft: #E8F3F1;
            --clinical-mint: #66AFA3;
            --clinical-amber: #C58A32;
            --clinical-red: #C65A5A;
        }

        html, body, .stApp {
            background-color: var(--clinical-bg) !important;
            color: var(--clinical-text) !important;
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        }

        .block-container {
            max-width: 1380px;
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
        }

        h1, .page-header {
            color: var(--clinical-text) !important;
            font-size: clamp(1.75rem, 2vw, 2.25rem) !important;
            font-weight: 700 !important;
            letter-spacing: -0.04em !important;
            margin-bottom: 1.25rem !important;
        }

        h2, h3, .section-header {
            color: var(--clinical-text) !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em !important;
        }

        [data-testid="stSidebar"] {
            background: #FBFCFC !important;
            border-right: 1px solid var(--clinical-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            width: 248px;
            padding: 1.25rem 1rem;
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--clinical-border);
            margin: 1rem 0;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
            gap: 7px;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
            width: 100% !important;
            min-height: 44px;
            border: 1px solid transparent;
            border-radius: 12px;
            padding: 10px 12px !important;
            background: transparent;
            transition: 160ms ease;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
            background: #F1F6F5;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
            background: var(--clinical-primary-soft) !important;
            border-color: #CDE0DC !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] p,
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
            color: var(--clinical-primary) !important;
            font-weight: 700 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }

        .card {
            min-height: 132px;
            border: 1px solid var(--clinical-border);
            border-radius: 16px;
            padding: 18px;
            background: var(--clinical-surface);
            box-shadow: 0 8px 24px rgba(48, 91, 84, 0.055);
        }

        .card .label {
            color: var(--clinical-muted);
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .card .value {
            color: var(--clinical-text);
            font-size: clamp(1.8rem, 3vw, 2.55rem);
            font-weight: 700;
            letter-spacing: -0.04em;
            line-height: 1.05;
            margin-top: 8px;
        }

        .card .sub {
            color: var(--clinical-muted);
            font-size: 0.78rem;
            margin-top: 8px;
        }

        .badge {
            border-radius: 14px;
            padding: 13px 16px;
            box-shadow: none;
            color: #FFFFFF;
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 14px;
            text-align: center;
        }

        .countdown-banner {
            background: #FFF5F4;
            border: 1px solid #E8BDB8;
            border-radius: 14px;
            color: #964A43;
            margin-bottom: 12px;
            padding: 14px 18px;
            text-align: center;
            font-weight: 700;
        }

        .info-pill {
            display: inline-block;
            border-radius: 999px;
            background: var(--clinical-primary-soft);
            color: var(--clinical-primary);
            font-size: 0.78rem;
            font-weight: 600;
            padding: 5px 11px;
        }

        [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border: 1px solid var(--clinical-border) !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 24px rgba(48, 91, 84, 0.055) !important;
        }

        .stDataFrame {
            background: #FFFFFF;
            border: 1px solid var(--clinical-border);
            border-radius: 12px;
        }

        [data-testid="stButton"] button,
        [data-testid="stLinkButton"] a {
            border-radius: 11px !important;
            border-color: var(--clinical-border) !important;
            font-weight: 600;
            min-height: 42px;
        }

        [data-testid="stButton"] button[kind="primary"],
        [data-testid="stLinkButton"] a[kind="primary"] {
            background: var(--clinical-primary) !important;
            border-color: var(--clinical-primary) !important;
            color: #FFFFFF !important;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--clinical-border);
            border-radius: 14px;
            background: #FFFFFF;
            margin-bottom: 8px;
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 14px !important;
            border-color: var(--clinical-border) !important;
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
        }

        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid var(--clinical-border);
            border-radius: 16px;
            padding: 16px;
        }

        [data-baseweb="slider"] [role="slider"] {
            background: var(--clinical-primary) !important;
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            border-radius: 10px !important;
        }

        .driveguard-brand {
            display: flex;
            align-items: center;
            gap: .5rem;
            color: var(--clinical-primary);
            font-size: 1.3rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .driveguard-brand span { color: var(--clinical-text); }
        .card { border-radius: 10px; box-shadow: none; }
        [data-testid="stImage"] img { border-radius: 8px; }
        [data-testid="stHeader"] { background: var(--clinical-bg); }
        @media (max-width: 900px) {
            .block-container { padding: 1.5rem 1rem 3rem !important; }
        }

        @media (max-width: 900px) {
            [data-testid="stSidebar"] > div:first-child {
                width: 100%;
            }
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(label: str, color: str) -> str:
    return (
        f'<div class="badge" style="background:{color}20;color:#18312e;'
        f'border:1px solid {color};border-left:5px solid {color};">{label}</div>'
    )


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


BROWSER_GEOLOCATION_EXPRESSION = """new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
        if (!settled) {
            settled = true;
            resolve(value);
        }
    };
    const timeoutId = setTimeout(
        () => finish({error: {code: 3, message: "Geolocation request timed out."}}),
        12000
    );
    if (!navigator.geolocation) {
        clearTimeout(timeoutId);
        finish({error: {code: 0, message: "Geolocation is not supported by this browser."}});
        return;
    }
    try {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                clearTimeout(timeoutId);
                finish({
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude,
                    accuracy: position.coords.accuracy,
                    timestamp: position.timestamp
                });
            },
            (error) => {
                clearTimeout(timeoutId);
                finish({error: {code: error.code, message: error.message}});
            },
            {enableHighAccuracy: true, timeout: 10000, maximumAge: 0}
        );
    } catch (error) {
        clearTimeout(timeoutId);
        finish({error: {code: 0, message: String(error)}});
    }
})"""


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "ear_threshold": config.EAR_THRESHOLD,
        "mar_threshold": config.MAR_THRESHOLD,
        "consec_frames": config.CONSEC_FRAMES_THRESHOLD,
        "logger": None,
        "last_summary": None,
        "last_csv": None,
        "last_csv_name": None,
        "driver_name": "Driver",
        "emergency_dry_run": config.EMERGENCY_DRY_RUN_DEFAULT,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if "chart_buf" not in st.session_state:
        st.session_state.chart_buf = deque(maxlen=config.CHART_HISTORY_FRAMES)

    if "contacts_book" not in st.session_state:
        st.session_state.contacts_book = ContactBook(config.CONTACTS_FILE)

    if "location_provider" not in st.session_state:
        st.session_state.location_provider = LocationProvider()

    if "sleep_advisor" not in st.session_state:
        st.session_state.sleep_advisor = SleepAdvisor()

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = FatigueChatBot(advisor=st.session_state.sleep_advisor)

    if "emergency_mgr" not in st.session_state:
        def get_snapshot():
            frame = st.session_state.get("latest_frame")
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.SNAPSHOT_JPEG_QUALITY])
                if ok:
                    return buf.tobytes()
            return None

        st.session_state.emergency_mgr = EmergencyManager(
            contact_book=st.session_state.contacts_book,
            location_provider=st.session_state.location_provider,
            snapshot_provider=get_snapshot,
            driver_name=st.session_state.driver_name,
            dry_run=st.session_state.emergency_dry_run,
        )

    pending = st.session_state.pop("pending_thresholds", None)
    if pending is not None:
        st.session_state["ear_threshold"] = round(pending[0], 2)
        st.session_state["mar_threshold"] = round(pending[1], 2)


def start_session() -> None:
    st.session_state.logger = SessionLogger()
    st.session_state.last_summary = None
    st.session_state.last_csv = None
    st.session_state.last_csv_name = None
    st.session_state.chart_buf.clear()
    st.session_state.emergency_mgr.reset()


def close_session() -> None:
    logger = st.session_state.logger
    if logger is None:
        return
    logger.close()
    st.session_state.last_summary = logger.summary_text()
    st.session_state.last_csv = logger.read_bytes()
    st.session_state.last_csv_name = logger.path.name
    st.session_state.logger = None


def finish_calibration(result: CalibrationResult, logger: SessionLogger, ui: dict) -> None:
    logger.log_calibration(result)
    st.session_state["calib_message"] = (result.success, result.message)
    if result.success:
        st.session_state["pending_thresholds"] = (result.ear_threshold, result.mar_threshold)
    ui["calib"].empty()
    st.rerun()


# ---------------------------------------------------------------------------
# Camera & Mesh Lifecycle
# ---------------------------------------------------------------------------
def get_camera(index: int):
    cam = st.session_state.get("camera")
    if cam is not None and st.session_state.get("camera_open_index") == index and cam.isOpened():
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
# Tab 1: Live Monitor
# ---------------------------------------------------------------------------
def render_live_ui(
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
    emergency_status,
) -> None:
    ui["video"].image(frame, channels="BGR")
    ui["badge"].markdown(badge_html(label, color), unsafe_allow_html=True)

    # Emergency Countdown Warning Banner
    if emergency_status.phase == PHASE_COUNTDOWN:
        ui["alert_banner"].markdown(
            f'<div class="countdown-banner">Emergency countdown · {int(emergency_status.countdown_remaining)} seconds remaining<br>'
            f'<span>Cancel immediately if the driver is awake</span></div>',
            unsafe_allow_html=True,
        )
    elif emergency_status.phase == PHASE_SENDING:
        ui["alert_banner"].warning("📤 Escalation in progress: Dispatching alert to emergency contacts...")
    elif emergency_status.phase == PHASE_COOLDOWN:
        ui["alert_banner"].info(f"⏳ Escalation cooldown active ({int(emergency_status.cooldown_remaining)}s left)")
    else:
        ui["alert_banner"].empty()

    ear_text = "--" if ear is None else f"{ear:.2f}"
    mar_text = "--" if mar is None else f"{mar:.2f}"
    ear_accent = config.STATE_COLORS[config.STATE_CRITICAL_DROWSY] if result.eyes_closed else config.STATE_COLORS[config.STATE_ALERT]
    mar_accent = config.STATE_COLORS[config.STATE_MILD_FATIGUE] if result.yawning else config.STATE_COLORS[config.STATE_ALERT]

    ui["ear"].markdown(
        card_html("Eye aspect ratio", ear_text, f"closed below {settings['ear_threshold']:.2f}", ear_accent),
        unsafe_allow_html=True,
    )
    ui["mar"].markdown(
        card_html("Mouth aspect ratio", mar_text, f"yawn above {settings['mar_threshold']:.2f}", mar_accent),
        unsafe_allow_html=True,
    )
    ui["perclos"].markdown(
        card_html(
            "PERCLOS",
            f"{result.perclos * 100:.1f}%",
            f"window: {config.PERCLOS_WINDOW_SIZE} frames",
            perclos_color(result.perclos, config.PERCLOS_MILD_THRESHOLD, config.PERCLOS_CRITICAL_THRESHOLD),
        ),
        unsafe_allow_html=True,
    )
    ui["yawns"].markdown(
        card_html(
            "Yawns detected",
            str(logger.yawn_total),
            f"warnings: {logger.warning_count}, critical: {logger.critical_count}",
            config.STATE_COLORS[config.STATE_ALERT],
        ),
        unsafe_allow_html=True,
    )
    ui["info"].caption(f"{fps:.1f} FPS | Escalation Mode: {'Dry Run' if emergency_status.report and emergency_status.report.dry_run else 'Active'}")


def render_chart(slot, buffer: deque) -> None:
    if not buffer:
        return
    data = pd.DataFrame(list(buffer), columns=["EAR", "PERCLOS", "EAR threshold"])
    slot.line_chart(data, height=220)


def run_monitor(settings: dict, logger: SessionLogger, ui: dict) -> None:
    cap = get_camera(settings["camera_index"])
    if cap is None:
        st.error(f"Could not open camera {settings['camera_index']}. Close other apps or check permissions.")
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
    emergency_mgr: EmergencyManager = st.session_state.emergency_mgr

    if settings["calibrate"]:
        calibrator.start()

    frame_count = 0
    fps = 0.0
    last_tick = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            release_camera()
            st.error("Lost the camera feed.")
            break

        frame = cv2.flip(frame, 1)
        frame, ear, mar = extractor.process_frame(frame)
        st.session_state.latest_frame = frame
        frame_count += 1
        refresh = frame_count % UI_UPDATE_EVERY == 0

        # Calibration
        if calibrator.active:
            if calibrator.add_sample(ear, mar):
                finish_calibration(calibrator.result, logger, ui)
            cv2.putText(
                frame,
                "CALIBRATING: eyes open, mouth closed",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                hex_to_bgr(CALIBRATION_COLOR),
                2,
                cv2.LINE_AA,
            )
            if refresh:
                remaining = config.CALIBRATION_DURATION_SEC * (1.0 - calibrator.progress())
                ui["video"].image(frame, channels="BGR")
                ui["badge"].markdown(badge_html("CALIBRATING", CALIBRATION_COLOR), unsafe_allow_html=True)
                ui["calib"].progress(
                    calibrator.progress(),
                    text=f"Look at the camera and hold still: {remaining:.0f}s left",
                )
            continue

        # Analytics & Emergency Escalation
        result: AnalyticsResult = analytics.update(ear, mar)
        logger.update(result, ear, mar)
        sound.update(result.state, ui["sound"])

        emergency_status = emergency_mgr.update(result.state, result.face_detected)

        chart_buffer.append(
            (
                np.nan if ear is None else ear,
                result.perclos,
                settings["ear_threshold"],
            )
        )

        if result.face_detected:
            label = STATE_LABELS[result.state]
            color = config.STATE_COLORS[result.state]
        else:
            label, color = "NO FACE DETECTED", NO_FACE_COLOR

        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            hex_to_bgr(color),
            2,
            cv2.LINE_AA,
        )

        if refresh:
            now = time.time()
            fps = UI_UPDATE_EVERY / max(now - last_tick, 1e-6)
            last_tick = now
            render_live_ui(ui, frame, result, ear, mar, label, color, fps, settings, logger, emergency_status)

        if frame_count % config.CHART_UPDATE_EVERY == 0:
            render_chart(ui["chart"], chart_buffer)


# ---------------------------------------------------------------------------
# Tab 2: Travel Readiness & Sleep Advisor
# ---------------------------------------------------------------------------
def render_sleep_advisor_tab() -> None:
    st.markdown(
        '<p style="color:#718783;font-size:.72rem;font-weight:700;letter-spacing:.12em;margin-bottom:.35rem;">TRAVEL HEALTH</p>'
        '<h1 class="page-header">Travel Readiness</h1>',
        unsafe_allow_html=True,
    )
    advisor: SleepAdvisor = st.session_state.sleep_advisor

    form_col, results_col = st.columns([1, 1.3])
    # Compact Sleep Logging Form inside its own card
    with form_col, st.container(border=True):
        st.markdown('<h3 class="section-header">📝 Log Sleep Period</h3>', unsafe_allow_html=True)
        with st.form("sleep_log_form", clear_on_submit=True):
            fcol1, fcol2, fcol3 = st.columns(3)
            today = dt.date.today()
            log_date = fcol1.date_input("Wake-up Date", value=today, max_value=today)
            bed_time = fcol2.time_input("Bed Time", value=dt.time(23, 0))
            wake_time = fcol3.time_input("Wake Time", value=dt.time(7, 0))

            qcol1, qcol2 = st.columns([1, 2])
            quality = qcol1.selectbox("Sleep Quality", ["good", "fair", "poor"], index=0)
            notes = qcol2.text_input("Notes (optional)", placeholder="e.g. Woke up twice, took coffee before bed")

            submitted = st.form_submit_button("Save Sleep Entry", type="primary", use_container_width=True)
            if submitted:
                try:
                    date_str = log_date.strftime("%Y-%m-%d")
                    bed_str = bed_time.strftime("%H:%M")
                    wake_str = wake_time.strftime("%H:%M")
                    advisor.log_sleep(date_str, bed_str, wake_str, quality=quality, notes=notes)
                    st.success(f"Sleep entry saved for {date_str}!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error saving entry: {exc}")

    readiness = advisor.check_readiness()
    with results_col, st.container(border=True):
        st.subheader("Readiness estimate")
        st.metric("Readiness score", f"{readiness.score}/100")
        st.progress(min(100, max(0, readiness.score)))
        st.caption("This estimate can’t determine whether it’s safe to drive.")
        if readiness.last_night_hours is not None:
            st.write(f"Most recent sleep: {readiness.last_night_hours:.1f} hours")
        with st.expander("Score details and guidance"):
            st.write(f"Sleep debt: {readiness.sleep_debt_hours:.1f} hours")
            st.write(f"Time awake: {readiness.hours_awake:.1f} hours")
            for penalty in readiness.penalties:
                st.write(f"-{penalty.points_lost} points · {penalty.description}")
            for tip in readiness.advice:
                st.info(tip)

    # Past Sleep History Table inside its own card
    entries = advisor.log_book.entries
    with results_col, st.container(border=True):
        st.markdown('<h3 class="section-header">Recent sleep</h3>', unsafe_allow_html=True)
        if entries:
            records = [
                {
                    "Date": e.date,
                    "Bed Time": e.bed_time,
                    "Wake Time": e.wake_time,
                    "Duration (hrs)": e.duration_hours,
                    "Quality": e.quality.upper(),
                    "Notes": e.notes,
                }
                for e in entries
            ]
            st.dataframe(pd.DataFrame(records), use_container_width=True)

            dcol1, dcol2 = st.columns([3, 1])
            date_to_remove = dcol1.selectbox("Select date to remove", [e.date for e in entries], label_visibility="collapsed")
            if dcol2.button("Delete Entry", type="secondary", use_container_width=True):
                if advisor.log_book.remove_entry(date_to_remove):
                    st.success(f"Removed entry for {date_to_remove}")
                    st.rerun()
        else:
            st.caption("No sleep history logged yet. Use the form above to add your recent sleep.")


# ---------------------------------------------------------------------------
# Tab 3: Emergency Contacts & Location
# ---------------------------------------------------------------------------
def render_emergency_tab(request_fresh_location: bool = False) -> None:
    st.markdown(
        '<p style="color:#718783;font-size:.72rem;font-weight:700;letter-spacing:.12em;margin-bottom:.35rem;">ESCALATION</p>'
        '<h1 class="page-header">Emergency</h1>',
        unsafe_allow_html=True,
    )
    contacts_book: ContactBook = st.session_state.contacts_book
    emergency_mgr: EmergencyManager = st.session_state.emergency_mgr
    loc_provider: LocationProvider = st.session_state.location_provider

    if request_fresh_location:
        _start_browser_location_request()

    pending_request_id = st.session_state.get("location_request_pending")
    if pending_request_id is not None:
        browser_location = streamlit_js_eval(
            js_expressions=BROWSER_GEOLOCATION_EXPRESSION,
            key=f"browser_location_{pending_request_id}",
        )
        if isinstance(browser_location, dict):
            if "error" in browser_location:
                error = browser_location["error"]
                error_code = error.get("code") if isinstance(error, dict) else None
                loc_provider.update_from_browser(browser_location)
                st.session_state["geo_error"] = {
                    1: "denied",
                    3: "timeout",
                }.get(error_code, "unavailable")
                st.session_state.pop("location_request_pending", None)
            elif loc_provider.update_from_browser(browser_location):
                st.session_state.pop("geo_error", None)
                st.session_state.pop("location_request_pending", None)
            else:
                st.session_state["geo_error"] = "invalid"
                st.session_state.pop("location_request_pending", None)
        elif browser_location is not None:
            st.session_state["geo_error"] = "unavailable"
            st.session_state.pop("location_request_pending", None)

    geo_error = st.session_state.get("geo_error")
    current_loc = loc_provider.current_precise()


    # Configuration Banner inside a bordered card
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            st.session_state.driver_name = st.text_input("Driver Name", value=st.session_state.driver_name)
            emergency_mgr.driver_name = st.session_state.driver_name
        with c2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            dry_run = st.checkbox("Dry Run Mode (Simulate Alerts)", value=emergency_mgr.dry_run)
            emergency_mgr.dry_run = dry_run
            st.session_state.emergency_dry_run = dry_run
        with c3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            token_set = bool(os.environ.get(config.TELEGRAM_TOKEN_ENV))
            if token_set:
                st.success("Telegram Bot Token: Configured")
            else:
                st.warning("Telegram Bot Token: Not Set (Dry run)")

    # Show location above contact management.
    col_geo = st.container()
    col_contacts = st.container()

    with col_geo:
        with st.container(border=True):
            st.markdown('<h3 class="section-header">📍 Driver Geolocation</h3>', unsafe_allow_html=True)
            if geo_error == "denied":
                st.warning("Location permission was denied. Allow browser location access, then refresh.")
            elif geo_error == "timeout":
                st.warning("Browser location timed out. Check location services, then refresh.")
            elif geo_error == "unavailable":
                st.warning("Browser location is unavailable. No precise location will be sent.")
            elif geo_error == "invalid":
                st.warning("Browser returned invalid coordinates. No precise location will be sent.")

            if current_loc is not None:
                if current_loc.source == SOURCE_MANUAL:
                    source_badge = "Manual Override"
                elif current_loc.source == SOURCE_BROWSER:
                    source_badge = "Browser device location (GPS or Wi-Fi)"
                    if time.time() - current_loc.timestamp > BROWSER_MAX_AGE_SEC:
                        source_badge = "Browser GPS (stale fallback)"
                elif current_loc.source == SOURCE_IP:
                    source_badge = "IP Approximate"
                else:
                    source_badge = current_loc.source
                st.map(
                    pd.DataFrame({"lat": [current_loc.lat], "lon": [current_loc.lon]}),
                    zoom=14,
                )
                updated_at = dt.datetime.fromtimestamp(current_loc.timestamp).strftime("%H:%M:%S")
                st.caption(f"Location updated at {updated_at}")
                st.markdown(f"**Coordinates:** `{current_loc.lat:.5f}, {current_loc.lon:.5f}`")
                st.markdown(f"**Source:** `{source_badge}`")
                if current_loc.accuracy_m is not None:
                    st.markdown(f"**Accuracy:** `{current_loc.accuracy_m:.0f} m`")
                st.markdown(f"**Description:** {describe(current_loc)}")
                st.link_button("📍 Open in Google Maps", map_link(current_loc), use_container_width=True)
                st.caption("Verify this pin before monitoring. Browser location can be inaccurate on laptops even when reported accuracy is good.")
                if st.session_state.get("location_request_pending") is not None:
                    st.info("Refreshing your browser location...")
            else:
                if st.session_state.get("location_request_pending") is not None:
                    st.info("Requesting your current browser location...")
                elif not geo_error:
                    st.info("No recent accurate device fix. Press refresh to try again.")

            with st.expander("Pair your driver phone"):
                st.caption("Share your location directly with the bot, then discover and select your own account.")
                if st.button("Discover driver phone"):
                    chats, error = TelegramClient().location_chats()
                    st.session_state["driver_phone_choices"] = chats
                    if error:
                        st.warning(error)
                    elif not chats:
                        st.warning("No private location messages found.")
                choices = st.session_state.get("driver_phone_choices", [])
                if choices:
                    selected = st.selectbox("Select your own account", choices,
                        format_func=lambda chat: f"{chat.name} (…{chat.chat_id[-4:]})")
                    if st.button("Confirm driver phone"):
                        os.environ["DRIVER_LOCATION_CHAT_ID"] = selected.chat_id
                        st.success("Driver phone paired for this running application.")
            if st.button("Check Phone Live Location", use_container_width=True):
                phone_client = TelegramClient()
                phone_fix = phone_client.phone_location()
                if phone_fix is None:
                    st.warning(getattr(phone_client, "location_status", "Phone location unavailable. Restart Streamlit to reload the updated backend."))
                else:
                    st.success(getattr(phone_client, "location_status", "Phone location unavailable. Restart Streamlit to reload the updated backend."))
                    st.write(describe(phone_fix))
                    st.link_button("Verify phone pin", map_link(phone_fix))

            if st.button("🔄 Refresh Current Location", use_container_width=True):
                _start_browser_location_request()
                st.rerun()

            with st.expander("Manual Coordinate Override", expanded=False):
                st.caption("Manual coordinates are for testing only and are not used in emergency alerts.")
                default_lat = float(current_loc.lat) if current_loc else 0.0
                default_lon = float(current_loc.lon) if current_loc else 0.0
                man_lat = st.number_input("Latitude", value=default_lat, format="%.5f")
                man_lon = st.number_input("Longitude", value=default_lon, format="%.5f")
                mcol1, mcol2 = st.columns(2)
                if mcol1.button("Set Manual", use_container_width=True):
                    loc_provider.set_manual(man_lat, man_lon)
                    st.success("Manual coordinates applied.")
                    st.rerun()
                if mcol2.button("Clear Override", use_container_width=True):
                    loc_provider.clear_manual()
                    st.info("Cleared manual override. Reverting to device GPS.")
                    st.rerun()

    with col_contacts:
        with st.container(border=True):
            st.markdown(f'<h3 class="section-header">👥 Emergency Contacts ({len(contacts_book)} / {MAX_CONTACTS})</h3>', unsafe_allow_html=True)

            if contacts_book.contacts:
                contact_data = []
                for c in contacts_book.contacts:
                    contact_data.append({
                        "Name": c.name,
                        "Masked Chat ID": mask_chat_id(c.chat_id),
                        "Relationship": c.relation or "--",
                    })
                st.dataframe(pd.DataFrame(contact_data), use_container_width=True)

                rcol1, rcol2 = st.columns([2, 1])
                contact_options = {f"{c.name} ({mask_chat_id(c.chat_id)})": c.chat_id for c in contacts_book.contacts}
                selected_label = rcol1.selectbox("Select contact to delete", list(contact_options.keys()), label_visibility="collapsed")
                if rcol2.button("Delete Contact", type="secondary", use_container_width=True):
                    target_id = contact_options[selected_label]
                    contacts_book.remove(target_id)
                    st.success("Contact deleted.")
                    st.rerun()
            else:
                st.caption("No emergency contacts saved yet.")

            # Add Contact Expander
            if len(contacts_book) < MAX_CONTACTS:
                with st.expander("➕ Add New Emergency Contact"):
                    with st.form("add_contact_form", clear_on_submit=True):
                        c_name = st.text_input("Contact Name (e.g. Mom, Partner)")
                        c_id = st.text_input("Telegram Chat ID (Numeric)")
                        c_rel = st.text_input("Relationship (optional)")
                        add_sub = st.form_submit_button("Add Contact", type="primary", use_container_width=True)
                        if add_sub:
                            try:
                                contacts_book.add(c_name, c_id, relation=c_rel)
                                st.success(f"Added {c_name} to emergency contacts!")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not add contact: {exc}")

            st.divider()

            # Test Telegram Alert Action Box
            st.markdown('<h3 class="section-header">🚀 Test Telegram Escalation</h3>', unsafe_allow_html=True)
            if st.button("Send Test Emergency Alert Now", type="primary", use_container_width=True):
                client = TelegramClient()
                if not client.configured and not emergency_mgr.dry_run:
                    st.error("Telegram bot token is not set! Set TELEGRAM_BOT_TOKEN or enable Dry Run mode.")
                elif len(contacts_book) == 0:
                    st.error("Add at least one contact to test alert delivery.")
                else:
                    with st.spinner("Dispatching test alert..."):
                        test_text = (
                            f"TEST ALERT from Driver Drowsiness System.\n"
                            f"Driver: {emergency_mgr.driver_name}\n"
                            f"Status: System test message. Please ignore."
                        )
                        test_location = loc_provider.current_precise()
                        if test_location is not None:
                            test_text += f"\nLocation: {describe(test_location)}\n{map_link(test_location)}"
                        else:
                            test_text += "\nLocation: no recent accurate device fix available."
                        st.caption(test_text)
                        success_count = 0
                        for c in contacts_book.contacts:
                            if emergency_mgr.dry_run:
                                success_count += 1
                            else:
                                res = client.send_message(c.chat_id, test_text)
                                if res.ok:
                                    success_count += 1
                                else:
                                    st.warning(f"Failed delivery to {c.name}: {res.error}")
                        if emergency_mgr.dry_run:
                            st.info(f"[Dry Run] Successfully simulated delivery to {success_count} contacts.")
                        else:
                            st.success(f"Delivered test alert to {success_count} of {len(contacts_book)} contacts.")


def _start_browser_location_request() -> None:
    request_id = st.session_state.get("location_request_id", 0) + 1
    st.session_state["location_request_id"] = request_id
    st.session_state["location_request_pending"] = request_id
    st.session_state.pop("geo_error", None)


# ---------------------------------------------------------------------------
# Tab 4: Driver Fatigue Assistant Chatbot
# ---------------------------------------------------------------------------
def render_chatbot_tab() -> None:
    st.title("Sleep assistant")
    st.caption("Ask about sleep, fatigue, and driving.")
    bot: FatigueChatBot = st.session_state.chatbot
    suggestions, conversation = st.columns([1, 2])
    with suggestions, st.container(border=True):
        st.subheader("Suggested questions")
        for idx, prompt in enumerate(SUGGESTED_PROMPTS[:6]):
            if st.button(prompt, key=f"preset_{idx}", use_container_width=True):
                bot.chat(prompt)
                st.rerun()

    with conversation, st.container(border=True):
        st.caption("Online assistant" if bot.is_online else "Offline sleep guidance")
        if st.button("Clear chat"):
            bot.clear_history()
            st.rerun()
        for message in bot.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        user_query = st.chat_input("Ask a question about sleep or driving fatigue")
        if user_query:
            with st.spinner("Preparing a reply…"):
                bot.chat(user_query)
            st.rerun()


# ---------------------------------------------------------------------------
# Main Page Router
# ---------------------------------------------------------------------------
def select_page(page: str) -> None:
    st.session_state.workspace_page = page


def render_navigation(selected: str) -> None:
    top_navigation = selected in ("Emergency", "Sleep Assistant")
    if top_navigation:
        st.markdown(
            '<style>[data-testid="stSidebar"], '
            '[data-testid="stSidebarCollapsedControl"] {display:none}</style>',
            unsafe_allow_html=True,
        )
        brand_col, *links = st.columns([1.35, 1, 1, 1.2, 1.2])
        brand_col.markdown(BRAND, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(BRAND, unsafe_allow_html=True)
        links = [st.sidebar] * len(PAGES)
    for container, page in zip(links, PAGES):
        container.button(
            page,
            key=f"nav_{'top' if top_navigation else 'side'}_{page}",
            type="primary" if page == selected else "secondary",
            use_container_width=True,
            on_click=select_page,
            args=(page,),
        )


def render_monitor_tab() -> None:
    title, toggle = st.columns([3, 1])
    title.title("Driver monitoring")
    running = toggle.toggle("Monitoring", key="monitoring_on")
    calib_message = st.session_state.pop("calib_message", None)
    if calib_message:
        ok, message = calib_message
        (st.success if ok else st.warning)(message)

    alert_ph = st.empty()
    calib_ph = st.empty()
    camera, detection = st.columns([1.75, 1])
    with camera, st.container(border=True):
        st.subheader("Driver camera")
        video_ph = st.empty()
        info_ph = st.empty()
    with detection, st.container(border=True):
        st.subheader("Detection settings")
        ear_threshold = st.slider(
            "EAR threshold", config.EAR_THRESHOLD_MIN, config.EAR_THRESHOLD_MAX,
            step=.01, key="ear_threshold",
        )
        mar_threshold = st.slider(
            "MAR threshold", config.MAR_THRESHOLD_MIN, config.MAR_THRESHOLD_MAX,
            step=.01, key="mar_threshold",
        )
        calibrate = st.button("Calibrate thresholds", type="primary", disabled=not running)
        st.caption(f"While parked, keep your eyes open and mouth closed for {config.CALIBRATION_DURATION_SEC}s.")
        with st.expander("Camera and alert settings"):
            camera_index = st.number_input("Camera index", 0, 10, config.CAMERA_INDEX)
            full_mesh = st.checkbox("Draw full face mesh")
            consec_frames = st.slider("Consecutive frames", 5, 60, key="consec_frames")
            sound_on = st.checkbox("Sound alert on critical drowsiness", value=True)
    perclos_col, ear_col, status_col = st.columns([1, 1, 1.6])
    perclos_ph = perclos_col.empty()
    ear_ph = ear_col.empty()
    status_col.subheader("Fatigue status")
    badge_ph = status_col.empty()
    with st.expander("Additional telemetry"):
        mar_col, yawns_col = st.columns(2)
        mar_ph = mar_col.empty()
        yawns_ph = yawns_col.empty()
    with st.expander("Camera & sound check"):
        if st.button("Check camera", disabled=running):
            camera_available = get_camera(int(camera_index)) is not None
            release_camera()
            if camera_available:
                st.success("Camera available.")
            else:
                st.warning("Camera unavailable. Check the camera index and permissions.")
        st.audio(generate_beep_wav(), format="audio/wav")
    sound_ph = st.empty()
    with st.container(border=True):
        st.subheader("Session timeline")
        chart_ph = st.empty()

    if st.sidebar.button("I'm awake · Cancel alert", type="primary", use_container_width=True):
        canceled = st.session_state.emergency_mgr.cancel()
        (st.success if canceled else st.info)(
            "Emergency countdown cancelled." if canceled else "No active emergency countdown."
        )

    if running:
        if st.session_state.logger is None:
            start_session()
        settings = {
            "camera_index": int(camera_index), "full_mesh": full_mesh,
            "ear_threshold": ear_threshold, "mar_threshold": mar_threshold,
            "consec_frames": consec_frames, "sound": sound_on, "calibrate": calibrate,
        }
        ui = {
            "alert_banner": alert_ph, "badge": badge_ph, "calib": calib_ph,
            "video": video_ph, "info": info_ph, "ear": ear_ph, "mar": mar_ph,
            "perclos": perclos_ph, "yawns": yawns_ph, "chart": chart_ph, "sound": sound_ph,
        }
        run_monitor(settings, st.session_state.logger, ui)
        return

    if st.session_state.logger is not None:
        close_session()
    release_camera()
    release_extractor()
    perclos_ph.metric("PERCLOS", "—")
    ear_ph.metric("EAR", "—")
    badge_ph.caption("Monitoring is off")
    video_ph.info("Turn on Monitoring to start the camera.")
    chart_ph.caption("Live EAR and PERCLOS trends appear while monitoring.")
    if st.session_state.last_summary:
        with st.expander("Last drive summary"):
            st.text(st.session_state.last_summary)
            st.download_button(
                "Download CSV log", data=st.session_state.last_csv,
                file_name=st.session_state.last_csv_name, mime="text/csv",
            )


def main() -> None:
    st.set_page_config(page_title="DriveGuard", page_icon="🚗", layout="wide")
    inject_css()
    init_session_state()
    selected = st.session_state.get("workspace_page", "Monitoring")
    if selected not in PAGES:
        selected = "Monitoring"
    previous_page = st.session_state.get("_last_selected_page")
    st.session_state["_last_selected_page"] = selected
    render_navigation(selected)

    if selected == "Monitoring":
        render_monitor_tab()
        return
    for key in ("ear_threshold", "mar_threshold", "consec_frames"):
        # Keep detection settings when Streamlit removes the page's widgets.
        st.session_state[key] = st.session_state[key]
    if st.session_state.logger is not None:
        close_session()
    release_camera()
    release_extractor()

    if selected == "Travel Readiness":
        render_sleep_advisor_tab()
    elif selected == "Emergency":
        render_emergency_tab(request_fresh_location=previous_page != selected)
    else:
        render_chatbot_tab()


if __name__ == "__main__":
    main()
