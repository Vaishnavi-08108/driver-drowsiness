# Driver drowsiness detection system

A real-time driver drowsiness monitor built with Python, MediaPipe Face Mesh, OpenCV, and Streamlit. It reads a webcam feed, measures how open the driver's eyes and mouth are, and classifies the driver as alert, mildly fatigued, or critically drowsy. A browser dashboard shows the live video, the measurements, and the current state.

This is a mini-project for learning computer vision and temporal analysis. It is not a certified safety system and should not be relied on while driving.

## Features

- Face landmark tracking with MediaPipe Face Mesh (468 landmarks)
- Eye Aspect Ratio (EAR) for eye closure and Mouth Aspect Ratio (MAR) for yawning
- PERCLOS, the share of recent frames in which the eyes were closed
- Three-state classifier: ALERT, MILD_FATIGUE, CRITICAL_DROWSY
- Streamlit dashboard with a color-coded status badge and metric cards
- Sound alert in the browser while the driver is critically drowsy
- Auto-calibration of the EAR and MAR thresholds for the current person
- Live chart of EAR, PERCLOS, and the EAR threshold
- CSV event log for every session, with a summary and a download button

## How it works

Each frame goes through three stages.

1. `detector.py` finds the face and computes EAR and MAR from landmark distances.
2. `analytics.py` keeps a rolling history and decides the driver state.
3. `app.py` shows the result and triggers the sound alert and the log.

### Aspect ratios

Both ratios are the mean vertical opening divided by the horizontal width.

```
EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
MAR = (|p2 - p8| + |p3 - p7| + |p4 - p6|) / (3 * |p1 - p5|)
```

EAR falls toward 0 when the eye closes. MAR rises when the mouth opens wide. Open eyes usually read around 0.25 to 0.35, but the numbers vary between people, which is why calibration exists.

### PERCLOS

PERCLOS is the fraction of frames in a sliding window (300 frames by default, about 10 seconds at 30 FPS) in which EAR was below the threshold. It catches slow fatigue, where the driver blinks often and for longer without holding the eyes shut. It reads 0 until the window holds a tenth of its frames, so startup noise cannot trigger an alarm.

### States

| State | Condition | Badge |
|---|---|---|
| ALERT | None of the conditions below | Green |
| MILD_FATIGUE | PERCLOS at or above 0.15, or a yawn held for 15 frames | Yellow |
| CRITICAL_DROWSY | Eyes closed for 15 frames in a row, or PERCLOS at or above 0.30 | Red |

The most severe matching condition wins. A frame with no detected face is treated as missing data. It is never counted as a closed eye, and the badge turns grey.

## Project structure

```
driver-drowsiness-system/
├── requirements.txt
├── README.md
├── logs/                      created automatically
└── src/
    ├── __init__.py
    ├── config.py              all constants and thresholds
    ├── detector.py            FaceFeatureExtractor (MediaPipe, EAR, MAR)
    ├── analytics.py           DrowsinessAnalytics (PERCLOS, state machine)
    ├── alerts.py              SoundAlert (browser beep)
    ├── calibration.py         Calibrator (personal thresholds)
    ├── session_log.py         SessionLogger (CSV log and summary)
    └── app.py                 Streamlit dashboard
```

## Setup

Python 3.10 or 3.11 is recommended. MediaPipe's `solutions` API can misbehave on newer versions.

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux or macOS
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Check the install:

```bash
python -c "import cv2, mediapipe, numpy, streamlit; print(mediapipe.solutions.face_mesh)"
```

## Running

Run every command from the project root (`driver-drowsiness-system/`), not from inside `src/`. The imports depend on it.

```bash
streamlit run src/app.py
```

Streamlit opens the dashboard at `http://localhost:8501`.

## Using the dashboard

1. Turn on Start monitoring in the sidebar. The video, cards, and chart appear after a moment.
2. Click Calibrate my thresholds and look at the camera for 5 seconds with your eyes open and your mouth closed. The EAR and MAR sliders update to your own values. If calibration fails, for example because your face was hidden, the sliders stay as they were.
3. Close your eyes for about a second. The badge turns red and a beep plays. Open them and it recovers.
4. Open your mouth wide for about a second to trigger a yawn warning.
5. Turn off Start monitoring to see the session summary and download the CSV log.

Notes:

- Calibration works only while monitoring is on.
- Moving a slider restarts the analytics, so PERCLOS needs a few seconds to warm up again.
- Browsers allow sound only after you have clicked something on the page. Turning on Start monitoring counts.
- The sidebar values apply to the current session only. Copy your final numbers into `src/config.py` to make them permanent.

## Configuration

Everything lives in `src/config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `CAMERA_INDEX` | 0 | Which webcam to use |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | 640, 480 | Capture size |
| `EAR_THRESHOLD` | 0.22 | EAR below this counts as closed |
| `MAR_THRESHOLD` | 0.50 | MAR above this counts as a yawn |
| `CONSEC_FRAMES_THRESHOLD` | 15 | Frames a condition must hold |
| `PERCLOS_WINDOW_SIZE` | 300 | Rolling window length in frames |
| `PERCLOS_MILD_THRESHOLD` | 0.15 | PERCLOS cutoff for mild fatigue |
| `PERCLOS_CRITICAL_THRESHOLD` | 0.30 | PERCLOS cutoff for critical drowsiness |
| `CALIBRATION_DURATION_SEC` | 5 | Length of a calibration run |
| `EAR_CALIBRATION_RATIO` | 0.75 | New EAR threshold as a share of your open-eye EAR |
| `MAR_CALIBRATION_OFFSET` | 0.30 | Added to your closed-mouth MAR for the yawn threshold |
| `SOUND_COOLDOWN_SEC` | 2.0 | Minimum time between beeps |
| `CHART_HISTORY_FRAMES` | 300 | Frames shown in the live chart |

## Session logs

Each monitoring session writes `logs/session_<date>_<time>.csv`. A row is written per event, not per frame, so the file stays small.

| Column | Content |
|---|---|
| `timestamp` | Wall-clock time with milliseconds |
| `event` | `SESSION_START`, `STATE_CHANGE`, `YAWN`, `CALIBRATION_OK`, `CALIBRATION_FAILED`, `SESSION_END` |
| `state` | Driver state at the time of the event |
| `ear`, `mar` | Measurements (for calibration events, the thresholds now in use) |
| `perclos` | PERCLOS at the time of the event |
| `yawn_count` | Total yawns so far in the session |

The summary shows session length, warnings, critical alerts, yawns, peak PERCLOS, and the time spent in each state.

## Testing

The logic modules can be tested without a camera. Save each script in the project root and run it from there.

| Script | Checks |
|---|---|
| `scratch_analytics_test.py` | State machine, blinks, yawns, PERCLOS, missing face |
| `scratch_alerts_test.py` | Beep generation, cooldown, enable switch |
| `scratch_calibration_test.py` | Threshold maths, clamping, failure case |
| `scratch_session_log_test.py` | Event rows, alert counts, summary |
| `scratch_detector_test.py` | Needs a webcam. Shows the landmark overlay and live EAR and MAR |

Each camera-free script prints a "passed" line when every check succeeds.

## Troubleshooting

| Problem | Likely fix |
|---|---|
| `AttributeError` on `mp.solutions` | MediaPipe is too new. Reinstall with `pip install "mediapipe>=0.10.9,<0.10.22"` |
| Import errors from `cv2` | Both `opencv-python` and `opencv-contrib-python` are installed. Uninstall `opencv-python` |
| Crashes mentioning NumPy | Install NumPy 1.x: `pip install "numpy>=1.23,<2.0"` |
| `No module named src` | Run from the project root, and make sure `src/__init__.py` exists |
| Camera does not open | Close other apps that use the webcam, or change the camera index in the sidebar |
| No beep | Click anywhere on the page once, and check that the browser tab is not muted |
| Detection is jumpy | Improve the lighting, face the camera, and run the calibration again |
| Page freezes after a click | Stop the app and run `python -X faulthandler -m streamlit run src/app.py` to see where it fails |

The camera stays in use while monitoring is on. Turn monitoring off, or stop the app in the terminal, to release it.

## Limitations

- Accuracy depends on lighting, camera angle, and the person. Dim light, a strongly turned or tilted head, glasses glare, and thick frames can all disturb the eye landmarks.
- The thresholds are starting points, not validated values. Calibrate for each person.
- Only one face is tracked.
- The system has not been tested on real driving data.

## Optional hardware

Hardware alerts are not part of this version. `config.py` still holds the serial settings and the single-byte commands (`0` reset, `1` warning, `2` critical) in case you add an Arduino buzzer later. That needs `pip install "pyserial>=3.5,<4.0"` plus an Arduino sketch and serial code in `app.py`.

## Ideas for future work

- Head-pose and head-nod detection
- Night mode with infrared camera support
- Per-driver profiles that save calibration between sessions
- A trained classifier on top of the EAR, MAR, and PERCLOS features
