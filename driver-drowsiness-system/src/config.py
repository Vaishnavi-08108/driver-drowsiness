"""
config.py
Central configuration for the Driver Drowsiness Detection System.

Every tunable constant lives here so thresholds, camera settings, alert
settings, and logging settings can be changed in one place without touching
the logic in the other modules.
"""

import platform
from pathlib import Path

# ---------------------------------------------------------------------------
# Camera settings
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0          # 0 is usually the built-in webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
# Eye Aspect Ratio: values below this mean the eye is considered closed.
EAR_THRESHOLD = 0.22

# Mouth Aspect Ratio: values above this mean the mouth is considered open
# wide enough to count as a yawn.
MAR_THRESHOLD = 0.50

# Number of consecutive frames a condition must hold before it counts.
# At roughly 30 FPS, 15 frames is about half a second. This filters out
# normal blinks and single-frame landmark jitter.
CONSEC_FRAMES_THRESHOLD = 15

# ---------------------------------------------------------------------------
# PERCLOS (percentage of eye closure) settings
# ---------------------------------------------------------------------------
# Size of the sliding window in frames. 300 frames is about 10 seconds
# at 30 FPS.
PERCLOS_WINDOW_SIZE = 300

# PERCLOS cutoffs used by the state machine (fractions between 0 and 1).
# Values above the mild cutoff mean the eyes were closed for a noticeable
# share of the window; above the critical cutoff means severe drowsiness.
PERCLOS_MILD_THRESHOLD = 0.15
PERCLOS_CRITICAL_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh settings
# ---------------------------------------------------------------------------
MAX_NUM_FACES = 1                 # a driver monitor only needs one face
REFINE_LANDMARKS = True           # better accuracy around eyes and lips
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# NEW: Auto-calibration settings
# ---------------------------------------------------------------------------
# How long the driver looks at the camera (eyes open, mouth closed) while
# the system measures their personal baseline.
CALIBRATION_DURATION_SEC = 5

# Calibration fails if fewer valid face frames than this were collected,
# for example because the face was hidden for most of the 5 seconds.
CALIBRATION_MIN_SAMPLES = 30

# EAR threshold = median open-eye EAR * this ratio. A value of 0.75 puts the
# threshold at 75 percent of the normal open-eye reading.
EAR_CALIBRATION_RATIO = 0.75

# MAR threshold = median closed-mouth MAR + this offset.
MAR_CALIBRATION_OFFSET = 0.30

# Calibrated thresholds are clamped to these ranges, which match the
# dashboard sliders, so one bad measurement cannot disable detection.
EAR_THRESHOLD_MIN = 0.10
EAR_THRESHOLD_MAX = 0.40
MAR_THRESHOLD_MIN = 0.20
MAR_THRESHOLD_MAX = 1.00

# ---------------------------------------------------------------------------
# NEW: Sound alert settings
# ---------------------------------------------------------------------------
SOUND_FREQUENCY_HZ = 880          # beep pitch
SOUND_DURATION_SEC = 0.6          # length of one beep
SOUND_SAMPLE_RATE = 22050         # samples per second for the generated tone

# Minimum time between two beeps while the state stays critical, so the
# alert repeats without turning into one continuous noise.
SOUND_COOLDOWN_SEC = 2.0

# ---------------------------------------------------------------------------
# NEW: Live chart settings
# ---------------------------------------------------------------------------
# Number of recent frames kept for the EAR / PERCLOS chart.
CHART_HISTORY_FRAMES = 300

# Redraw the chart every N frames. Charts are expensive to render, so a
# higher number keeps the video smoother.
CHART_UPDATE_EVERY = 10

# ---------------------------------------------------------------------------
# NEW: Session logging settings
# ---------------------------------------------------------------------------
# Logs go in a "logs" folder at the project root (next to requirements.txt).
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# Column order of the CSV event log.
LOG_COLUMNS = [
    "timestamp",
    "event",
    "state",
    "ear",
    "mar",
    "perclos",
    "yawn_count",
]

# ---------------------------------------------------------------------------
# Serial / hardware settings (optional, only used if you add an Arduino)
# ---------------------------------------------------------------------------
# Default port depends on the operating system. Change SERIAL_PORT
# manually if your Arduino shows up on a different port.
if platform.system() == "Windows":
    SERIAL_PORT = "COM3"
else:
    SERIAL_PORT = "/dev/ttyUSB0"   # on macOS this is often /dev/tty.usbmodemXXXX

SERIAL_BAUD_RATE = 9600
SERIAL_TIMEOUT = 1                 # seconds

# Single-byte commands for an optional Arduino buzzer sketch
SERIAL_CMD_RESET = b"0"            # all outputs off
SERIAL_CMD_WARNING = b"1"          # warning LED on
SERIAL_CMD_CRITICAL = b"2"         # active buzzer on

# ---------------------------------------------------------------------------
# Driver states (shared names so every module spells them the same way)
# ---------------------------------------------------------------------------
STATE_ALERT = "ALERT"
STATE_MILD_FATIGUE = "MILD_FATIGUE"
STATE_CRITICAL_DROWSY = "CRITICAL_DROWSY"

# Dashboard badge colors for each state (hex values used in app.py)
STATE_COLORS = {
    STATE_ALERT: "#2ecc71",            # green
    STATE_MILD_FATIGUE: "#f1c40f",     # yellow
    STATE_CRITICAL_DROWSY: "#e74c3c",  # red
}