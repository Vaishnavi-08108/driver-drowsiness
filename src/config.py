"""
config.py
Central configuration for the Driver Drowsiness Detection System.

Every tunable constant lives here so thresholds, camera settings, and
hardware settings can be changed in one place without touching the logic
in detector.py, analytics.py, or app.py.
"""

import platform

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
# Values above MILD fatigue cutoff mean the eyes were closed for a
# noticeable share of the window; above CRITICAL means severe drowsiness.
PERCLOS_MI