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
from src.alerts import SoundAlert
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
            --clinical-bg: #F4F7F6;
            --clinical-surface: #FFFFFF;
            --clinical-text: #18312E;
            --clinical-muted: #718783;
            --clinical-border: #DDE7E4;
            --clinical-primary: #3A7D78;
            --clinical-primary-soft: #E8F3F1;
            --clinical-mint: #66AFA3;
            --clinical-amber: #C58A32;
            --clinical-red: #C65A5A;
        }

        html, body, .stApp, [class*="css"] {
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

    # Evaluate current readiness
    readiness = advisor.check_readiness()

    # 4 Top Metric Cards evenly across the top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="card" style="--accent:{readiness.color};">'
            f'<div class="label">Readiness Score</div>'
            f'<div class="value" style="color:{readiness.color};">{readiness.score}/100</div>'
            f'<div class="sub">Verdict: <b>{readiness.verdict}</b></div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        last_night = f"{readiness.last_night_hours:.1f}h" if readiness.last_night_hours is not None else "--"
        st.markdown(
            card_html("Last Night's Sleep", last_night, f"Target: {config.SLEEP_TARGET_HOURS:.1f}h", "#2563EB"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            card_html("Cumulative Debt", f"{readiness.sleep_debt_hours:.1f}h", f"Past {config.SLEEP_DEBT_DAYS} days", "#F59E0B"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            card_html("Hours Awake", f"{readiness.hours_awake:.1f}h", f"Limit: {config.AWAKE_FREE_HOURS:.0f}h free", "#8B5CF6"),
            unsafe_allow_html=True,
        )

    # Score Deductions & Action Plan side-by-side in two equal bordered cards
    col_deductions, col_plan = st.columns(2)
    with col_deductions:
        with st.container(border=True):
            st.markdown('<h3 class="section-header">⚠️ Score Deductions & Risk Factors</h3>', unsafe_allow_html=True)
            if readiness.penalties:
                for p in readiness.penalties:
                    st.error(f"**-{p.points_lost} pts** [{p.category}] — {p.description}")
            else:
                st.success("No penalties deducted. Sleep duration and circadian timing are optimal!")

    with col_plan:
        with st.container(border=True):
            st.markdown('<h3 class="section-header">💡 Personalized Action Plan</h3>', unsafe_allow_html=True)
            if readiness.advice:
                for tip in readiness.advice:
                    st.info(f"{tip}")
            else:
                st.success("You are in optimal shape to drive. Remember to take a break every 2 hours.")

    # Compact Sleep Logging Form inside its own card
    with st.container(border=True):
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

    # Past Sleep History Table inside its own card
    entries = advisor.log_book.entries
    with st.container(border=True):
        st.markdown('<h3 class="section-header">📋 Past Sleep Records</h3>', unsafe_allow_html=True)
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
        '<h1 class="page-header">Emergency Support</h1>',
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

    # 2-column split ([1, 1]): Left = Geolocation, Right = Contacts & Escalation
    col_geo, col_contacts = st.columns([1, 1])

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

            if st.button("Check Phone Live Location", use_container_width=True):
                phone_client = TelegramClient()
                phone_fix = phone_client.phone_location()
                if phone_fix is None:
                    st.warning(phone_client.location_status)
                else:
                    st.success(phone_client.location_status)
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
    # Centered container (max-width: 900px)
    _, center_col, _ = st.columns([1, 8, 1])
    with center_col:
        with st.container(border=True):
            st.markdown(
                '<p style="color:#718783;font-size:.72rem;font-weight:700;letter-spacing:.12em;margin-bottom:.35rem;">SLEEP SCIENCE</p>'
                '<h1 class="page-header">Fatigue Assistant</h1>',
                unsafe_allow_html=True,
            )
            bot: FatigueChatBot = st.session_state.chatbot

            col1, col2 = st.columns([3, 1])
            with col1:
                mode_label = "Online (Claude Messages API)" if bot.is_online else "Offline (Rule-Based Sleep Science Expert)"
                mode_color = "#10B981" if bot.is_online else "#F59E0B"
                st.markdown(f"**Mode:** <span style='color:{mode_color}; font-weight:700;'>{mode_label}</span>", unsafe_allow_html=True)
            with col2:
                if st.button("Clear Chat", use_container_width=True):
                    bot.clear_history()
                    st.rerun()

            # Preset suggestions
            st.markdown("<p style='font-size: 12.5px; font-weight: 600; color: #64748B; margin-top: 10px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;'>SUGGESTED QUESTIONS</p>", unsafe_allow_html=True)
            preset_cols = st.columns(3)
            for idx, prompt in enumerate(SUGGESTED_PROMPTS[:6]):
                col = preset_cols[idx % 3]
                if col.button(prompt, key=f"preset_{idx}", use_container_width=True):
                    bot.chat(prompt)
                    st.rerun()

            st.divider()

            # Render Chat History
            for message in bot.history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # User input
            user_query = st.chat_input("Ask a question about driving fatigue, naps, caffeine, or safety...")
            if user_query:
                with st.chat_message("user"):
                    st.markdown(user_query)
                with st.spinner("Consulting sleep science advisor..."):
                    reply = bot.chat(user_query)
                with st.chat_message("assistant"):
                    st.markdown(reply)
                st.rerun()


# ---------------------------------------------------------------------------
# Main Page Router
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Driver Drowsiness & Fatigue System",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_session_state()

    # -----------------------------------------------------------------------
    # Left Navigation & Sidebar Controls
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <div style="width:42px;height:42px;border-radius:14px;background:#E8F3F1;color:#3A7D78;display:flex;align-items:center;justify-content:center;font-size:1.15rem;font-weight:700;">SD</div>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #18312E; line-height: 1.1;">SafeDrive AI</div>
                    <div style="font-size: 0.72rem; color: #718783; font-weight: 600;">Driver safety</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Travel Readiness Status Pill in left sidebar
        try:
            r = st.session_state.sleep_advisor.check_readiness()
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid {r.color};
                            border-radius: 10px; padding: 10px 14px; margin: 10px 0 14px 0;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div style="font-size: 0.7rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Travel Readiness</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; display: flex; justify-content: space-between; align-items: center; margin-top: 2px;">
                        <span>{r.score}/100</span>
                        <span style="font-size: 0.8rem; padding: 2px 8px; border-radius: 10px; background: {r.color}20; color: {r.color}; font-weight: 700;">{r.verdict}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass

        # "I'm Awake" cancellation button in sidebar
        if st.button("⚡ I'M AWAKE (Cancel Alert)", type="primary", use_container_width=True):
            canceled = st.session_state.emergency_mgr.cancel()
            if canceled:
                st.success("Emergency countdown cancelled! Driver is alert.")
            else:
                st.info("No active emergency countdown.")

        st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin: 16px 0 4px 0; letter-spacing: 0.5px;'>NAVIGATION</p>", unsafe_allow_html=True)

        NAV_OPTIONS = [
            "Live Monitor",
            "Travel Readiness",
            "Emergency",
            "AI Assistant",
        ]
        selected_page = st.radio(
            "Navigation",
            NAV_OPTIONS,
            index=0,
            label_visibility="collapsed",
        )

        settings = None
        running = False
        calibrate_clicked = False

        # Live Monitor settings in sidebar when Live Monitor is selected
        if selected_page == "Live Monitor":
            st.divider()
            st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;'>MONITOR CONTROLS</p>", unsafe_allow_html=True)
            running = st.toggle("Start monitoring", value=False)

            st.subheader("Camera")
            camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=config.CAMERA_INDEX, step=1)
            full_mesh = st.checkbox("Draw full face mesh", value=False)

            st.subheader("Detection Thresholds")
            ear_threshold = st.slider("EAR threshold", min_value=config.EAR_THRESHOLD_MIN, max_value=config.EAR_THRESHOLD_MAX, step=0.01, key="ear_threshold")
            mar_threshold = st.slider("MAR threshold", min_value=config.MAR_THRESHOLD_MIN, max_value=config.MAR_THRESHOLD_MAX, step=0.01, key="mar_threshold")
            consec_frames = st.slider("Consecutive frames", min_value=5, max_value=60, step=1, key="consec_frames")

            st.subheader("Calibration")
            st.caption(f"Look at camera eyes open and mouth closed for {config.CALIBRATION_DURATION_SEC}s.")
            calibrate_clicked = st.button("Calibrate my thresholds", use_container_width=True)
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

    # -----------------------------------------------------------------------
    # Main Page Router based on Left Sidebar Selection
    # -----------------------------------------------------------------------
    if selected_page == "Live Monitor":
        calib_message = st.session_state.pop("calib_message", None)

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

        st.markdown(
            '<p style="color:#718783;font-size:.72rem;font-weight:700;letter-spacing:.12em;margin-bottom:.35rem;">DRIVER SAFETY</p>'
            '<h1 class="page-header">Live Monitor</h1>',
            unsafe_allow_html=True,
        )

        # Compact, horizontal status bar at the top
        alert_ph = st.empty()
        badge_ph = st.empty()
        calib_ph = st.empty()

        # Balanced 2-column grid ([1.2, 1] ratio)
        left, right = st.columns([1.2, 1])
        with left:
            with st.container(border=True):
                st.markdown('<h3 class="section-header">📷 Driver Camera Feed</h3>', unsafe_allow_html=True)
                video_ph = st.empty()
                info_ph = st.empty()

        with right:
            with st.container(border=True):
                st.markdown('<h3 class="section-header">📊 Telemetry Metrics</h3>', unsafe_allow_html=True)
                row1 = st.columns(2)
                row2 = st.columns(2)
                ear_ph = row1[0].empty()
                mar_ph = row1[1].empty()
                perclos_ph = row2[0].empty()
                yawns_ph = row2[1].empty()

                st.markdown('<h3 class="section-header">📈 Live EAR & PERCLOS Trend</h3>', unsafe_allow_html=True)
                chart_ph = st.empty()
                sound_ph = st.empty()

        ui = {
            "alert_banner": alert_ph,
            "badge": badge_ph,
            "calib": calib_ph,
            "video": video_ph,
            "info": info_ph,
            "ear": ear_ph,
            "mar": mar_ph,
            "perclos": perclos_ph,
            "yawns": yawns_ph,
            "chart": chart_ph,
            "sound": sound_ph,
        }

        if running:
            run_monitor(settings, st.session_state.logger, ui)
        else:
            release_camera()
            release_extractor()
            badge_ph.markdown(badge_html("MONITORING STOPPED", NO_FACE_COLOR), unsafe_allow_html=True)
            video_ph.info("Turn on **Start monitoring** in the left sidebar to begin.")

            if st.session_state.last_summary:
                with st.container(border=True):
                    st.markdown('<h3 class="section-header">📋 Last Drive Summary</h3>', unsafe_allow_html=True)
                    st.text(st.session_state.last_summary)
                    st.download_button(
                        "Download CSV log",
                        data=st.session_state.last_csv,
                        file_name=st.session_state.last_csv_name,
                        mime="text/csv",
                        type="primary",
                    )

    elif selected_page == "Travel Readiness":
        render_sleep_advisor_tab()

    elif selected_page == "Emergency":
        previous_page = st.session_state.get("_last_selected_page")
        request_fresh_location = previous_page != "Emergency"
        render_emergency_tab(request_fresh_location=request_fresh_location)

    elif selected_page == "AI Assistant":
        render_chatbot_tab()

    st.session_state["_last_selected_page"] = selected_page


if __name__ == "__main__":
    main()
