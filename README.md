# Driver Drowsiness System

A small, experimental Python application that estimates eye closure from a webcam feed. It uses MediaPipe Face Mesh landmarks to calculate the eye aspect ratio (EAR) and mouth aspect ratio (MAR), and tracks closed-eye time with a rolling PERCLOS metric. A Streamlit interface displays measurements and an optional Arduino sketch can drive a buzzer.

> This is a learning prototype, not a certified driver-monitoring or safety system. Do not rely on it to prevent accidents or replace rest, safe driving, or vehicle safety equipment.

## Architecture

![System architecture](docs/architecture.png)

Camera frames are passed to MediaPipe for facial landmarks. `src/detector.py` calculates EAR and MAR, and `src/analytics.py` tracks the fraction of observed time with eyes closed. `src/app.py` presents a manual webcam capture dashboard. An optional serial-connected Arduino can receive `ALERT` and `CLEAR` commands to operate a buzzer.

## Requirements

- Python 3.10 through 3.12
- A webcam, if using live capture
- Windows, macOS, or Linux; MediaPipe wheel availability depends on the platform and Python version
- Arduino-compatible board and buzzer only if using the optional hardware alert

## Setup

Create and activate a virtual environment, then install the pinned dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead. To install the local package metadata for development, run `pip install -e .`.

## Run

From the project root:

```text
streamlit run src/app.py
```

Allow camera access when prompted, enable camera capture in the dashboard, and capture frames periodically. The dashboard calculates metrics for each captured frame; it does not continuously monitor in the background. Camera access and MediaPipe are loaded only when capture is requested.

## Configuration and Metrics

Adjust the constants in `src/config.py` for eye-closure EAR, PERCLOS alert fraction, and analysis window duration. EAR and MAR are dimensionless ratios derived from six ordered landmark points. PERCLOS is the estimated fraction of observed time in the rolling window for which EAR is below the closure threshold. It is an estimate based on sampled frames, not a diagnosis.

## Optional Buzzer

The sketch in `hardware/buzzer_alert/buzzer_alert.ino` listens at 9600 baud for newline-terminated `ALERT` and `CLEAR` messages and switches the buzzer on digital pin 8. Connect the buzzer and board according to the board manufacturer's electrical guidance. The Python dashboard currently reports the alert in the UI; serial transport is intentionally left optional and is not automatically opened.

## Tests

The metric tests do not require a webcam or MediaPipe:

```text
python -m unittest discover -s tests -v
```

## Data and Privacy

No sample clips are included. Place only user-provided local test data under `data/`; that directory is excluded from Git by default. Obtain consent before recording or analyzing anyone, and handle face video as sensitive personal data.