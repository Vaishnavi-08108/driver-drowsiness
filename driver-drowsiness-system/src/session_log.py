"""
session_log.py
CSV event log and session summary for the Driver Drowsiness Detection System.

SessionLogger writes one CSV file per session into the logs/ folder. A row
is written for each event, not for each frame, so a 30 minute session
produces a few dozen rows instead of tens of thousands.

Events logged:
    SESSION_START       when the logger is created.
    STATE_CHANGE        every time the driver state changes.
    YAWN                every time a new yawn is counted.
    CALIBRATION_OK      calibration finished (ear and mar hold the new
                        thresholds).
    CALIBRATION_FAILED  calibration failed (ear and mar hold the defaults).
    SESSION_END         when close() is called.

Warnings and critical alerts are counted when the state gets MORE severe
(ALERT to MILD_FATIGUE counts as a warning, anything to CRITICAL_DROWSY
counts as a critical alert). Dropping back from critical to mild does not
add another warning.

Every method that reads time accepts an optional `now` argument so the test
script can simulate a whole session without waiting.
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from src import config
from src.analytics import AnalyticsResult
from src.calibration import CalibrationResult

EVENT_SESSION_START = "SESSION_START"
EVENT_STATE_CHANGE = "STATE_CHANGE"
EVENT_YAWN = "YAWN"
EVENT_CALIBRATION_OK = "CALIBRATION_OK"
EVENT_CALIBRATION_FAILED = "CALIBRATION_FAILED"
EVENT_SESSION_END = "SESSION_END"

# Higher number means a more severe state
_SEVERITY = {
    config.STATE_ALERT: 0,
    config.STATE_MILD_FATIGUE: 1,
    config.STATE_CRITICAL_DROWSY: 2,
}


class SessionLogger:
    """Writes driver events to a CSV file and tracks session statistics."""

    def __init__(
        self,
        log_dir: Union[str, Path] = config.LOG_DIR,
        now: Optional[float] = None,
    ) -> None:
        """
        Args:
            log_dir: folder for the CSV file. It is created if missing.
            now: session start time in seconds. Leave as None to use the
                clock; tests pass a fake time.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self._new_log_path()

        # newline="" is required by the csv module to avoid blank lines
        # between rows on Windows.
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(config.LOG_COLUMNS)

        self._start = time.monotonic() if now is None else now
        self._end: Optional[float] = None
        self._last_update = self._start
        self._closed = False

        self._state = config.STATE_ALERT
        self._last_analytics_yawns = 0

        # Statistics for the summary
        self.yawn_total = 0
        self.warning_count = 0
        self.critical_count = 0
        self.peak_perclos = 0.0
        self.seconds_in_state: Dict[str, float] = {s: 0.0 for s in _SEVERITY}
        self.seconds_no_face = 0.0

        self.log_event(EVENT_SESSION_START, state=self._state)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def update(
        self,
        result: AnalyticsResult,
        ear: Optional[float],
        mar: Optional[float],
        now: Optional[float] = None,
    ) -> None:
        """
        Call once per frame with the analytics result.

        Frames with no face add to the no-face time but never change the
        state, so a driver leaving the frame is not logged as recovered.
        Time is credited to the state that was active before this frame, so
        the totals lag by one frame, which is negligible at 30 FPS.
        """
        if self._closed:
            return

        now = time.monotonic() if now is None else now
        dt = max(0.0, now - self._last_update)
        self._last_update = now

        if not result.face_detected:
            self.seconds_no_face += dt
            return

        self.seconds_in_state[self._state] += dt
        self.peak_perclos = max(self.peak_perclos, result.perclos)

        # The analytics object restarts its yawn count when the dashboard
        # recreates it (for example after a slider change). Only positive
        # changes are added, so the session total never goes backwards.
        delta = result.yawn_count - self._last_analytics_yawns
        self._last_analytics_yawns = result.yawn_count
        if delta > 0:
            self.yawn_total += delta
            self.log_event(
                EVENT_YAWN, state=result.state, ear=ear, mar=mar,
                perclos=result.perclos, yawn_count=self.yawn_total,
            )

        if result.state != self._state:
            if _SEVERITY[result.state] > _SEVERITY[self._state]:
                if result.state == config.STATE_CRITICAL_DROWSY:
                    self.critical_count += 1
                else:
                    self.warning_count += 1
            self._state = result.state
            self.log_event(
                EVENT_STATE_CHANGE, state=result.state, ear=ear, mar=mar,
                perclos=result.perclos, yawn_count=self.yawn_total,
            )

    def log_calibration(self, result: CalibrationResult) -> None:
        """
        Record a calibration run. The ear and mar columns hold the
        thresholds that are now in use (the new ones on success, the
        defaults on failure).
        """
        event = EVENT_CALIBRATION_OK if result.success else EVENT_CALIBRATION_FAILED
        self.log_event(
            event, ear=result.ear_threshold, mar=result.mar_threshold,
            yawn_count=self.yawn_total,
        )

    def log_event(
        self,
        event: str,
        state: Optional[str] = None,
        ear: Optional[float] = None,
        mar: Optional[float] = None,
        perclos: Optional[float] = None,
        yawn_count: Optional[int] = None,
    ) -> None:
        """Write one row. Missing values are left blank. The row is flushed
        at once, so the file is complete even if the app crashes."""
        if self._closed:
            return
        self._writer.writerow([
            self._timestamp(),
            event,
            state or "",
            "" if ear is None else f"{ear:.3f}",
            "" if mar is None else f"{mar:.3f}",
            "" if perclos is None else f"{perclos:.4f}",
            "" if yawn_count is None else yawn_count,
        ])
        self._file.flush()

    def summary(self, now: Optional[float] = None) -> dict:
        """Return the session statistics as a dictionary."""
        if self._end is not None:
            end = self._end
        else:
            end = time.monotonic() if now is None else now
        return {
            "duration_sec": end - self._start,
            "warnings": self.warning_count,
            "critical_alerts": self.critical_count,
            "yawns": self.yawn_total,
            "peak_perclos": self.peak_perclos,
            "seconds_alert": self.seconds_in_state[config.STATE_ALERT],
            "seconds_mild": self.seconds_in_state[config.STATE_MILD_FATIGUE],
            "seconds_critical": self.seconds_in_state[config.STATE_CRITICAL_DROWSY],
            "seconds_no_face": self.seconds_no_face,
            "log_file": str(self.path),
        }

    def summary_text(self, now: Optional[float] = None) -> str:
        """Return the summary as plain text for the dashboard."""
        s = self.summary(now)
        minutes, seconds = divmod(int(s["duration_sec"]), 60)
        return (
            f"Session length: {minutes}m {seconds:02d}s\n"
            f"Warnings (mild fatigue): {s['warnings']}\n"
            f"Critical alerts: {s['critical_alerts']}\n"
            f"Yawns: {s['yawns']}\n"
            f"Peak PERCLOS: {s['peak_perclos'] * 100:.1f}%\n"
            f"Time alert: {s['seconds_alert']:.0f}s, "
            f"mild: {s['seconds_mild']:.0f}s, "
            f"critical: {s['seconds_critical']:.0f}s, "
            f"no face: {s['seconds_no_face']:.0f}s\n"
            f"Log file: {s['log_file']}"
        )

    def read_bytes(self) -> bytes:
        """Return the CSV file contents, for the dashboard download button."""
        if not self._closed:
            self._file.flush()
        return self.path.read_bytes()

    def close(self, now: Optional[float] = None) -> None:
        """Write the SESSION_END row and close the file. Safe to call twice."""
        if self._closed:
            return
        self._end = time.monotonic() if now is None else now
        self.log_event(
            EVENT_SESSION_END, state=self._state,
            perclos=self.peak_perclos, yawn_count=self.yawn_total,
        )
        self._closed = True
        self._file.close()

    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    def _new_log_path(self) -> Path:
        """Build session_YYYYmmdd_HHMMSS.csv, adding a suffix if the name is
        taken (two sessions started within the same second)."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"session_{stamp}.csv"
        n = 1
        while path.exists():
            path = self.log_dir / f"session_{stamp}_{n}.csv"
            n += 1
        return path

    @staticmethod
    def _timestamp() -> str:
        """Wall-clock time with milliseconds, e.g. 2026-11-16 14:03:22.481."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]