"""
calibration.py
Per-driver auto-calibration of the EAR and MAR thresholds.

Eyes and mouths differ from person to person, so one fixed threshold is only
a rough guess. The driver looks at the camera for a few seconds with eyes
open and mouth closed, and Calibrator turns those readings into personal
thresholds:

    EAR threshold = median open-eye EAR * EAR_CALIBRATION_RATIO
    MAR threshold = median closed-mouth MAR + MAR_CALIBRATION_OFFSET

The median is used instead of the mean so that a few blinks or a small
mouth movement during the 5 seconds do not shift the result.

The class never reads the clock on its own unless it has to. Every method
accepts an optional `now` argument, which lets the test script run a full
calibration without waiting.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src import config


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of one calibration run."""

    success: bool
    ear_threshold: float     # new EAR threshold (or the default if it failed)
    mar_threshold: float     # new MAR threshold (or the default if it failed)
    ear_open: float          # median open-eye EAR that was measured
    mar_closed: float        # median closed-mouth MAR that was measured
    samples: int             # number of valid face frames collected
    clamped: bool            # True if a threshold hit its allowed limit
    message: str             # text to show the user


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class Calibrator:
    """Collects EAR and MAR samples and computes personal thresholds."""

    def __init__(
        self,
        duration: float = config.CALIBRATION_DURATION_SEC,
        min_samples: int = config.CALIBRATION_MIN_SAMPLES,
        ear_ratio: float = config.EAR_CALIBRATION_RATIO,
        mar_offset: float = config.MAR_CALIBRATION_OFFSET,
    ) -> None:
        self.duration = duration
        self.min_samples = min_samples
        self.ear_ratio = ear_ratio
        self.mar_offset = mar_offset

        self._ear_samples: List[float] = []
        self._mar_samples: List[float] = []
        self._start_time: Optional[float] = None
        self.active = False
        self.result: Optional[CalibrationResult] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def start(self, now: Optional[float] = None) -> None:
        """Begin a new calibration run and discard any earlier data."""
        self._ear_samples.clear()
        self._mar_samples.clear()
        self._start_time = time.monotonic() if now is None else now
        self.active = True
        self.result = None

    def cancel(self) -> None:
        """Stop without producing a result."""
        self.active = False
        self._start_time = None

    def progress(self, now: Optional[float] = None) -> float:
        """Fraction of the calibration time that has passed (0.0 to 1.0)."""
        if not self.active or self._start_time is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return _clamp((now - self._start_time) / self.duration, 0.0, 1.0)

    def add_sample(
        self,
        ear: Optional[float],
        mar: Optional[float],
        now: Optional[float] = None,
    ) -> bool:
        """
        Feed one frame's measurements.

        Frames with no face (ear or mar is None) are skipped but still count
        toward the elapsed time.

        Returns:
            True on the call that finishes the run. The outcome is then in
            self.result. Returns False while still collecting or when no
            calibration is active.
        """
        if not self.active:
            return False

        now = time.monotonic() if now is None else now

        if ear is not None and mar is not None:
            self._ear_samples.append(float(ear))
            self._mar_samples.append(float(mar))

        if self.progress(now) >= 1.0:
            self.result = self._compute_result()
            self.active = False
            return True
        return False

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    def _compute_result(self) -> CalibrationResult:
        n = len(self._ear_samples)

        if n < self.min_samples:
            return CalibrationResult(
                success=False,
                ear_threshold=config.EAR_THRESHOLD,
                mar_threshold=config.MAR_THRESHOLD,
                ear_open=0.0,
                mar_closed=0.0,
                samples=n,
                clamped=False,
                message=(
                    f"Calibration failed: only {n} usable frames "
                    f"(need {self.min_samples}). Face the camera in good "
                    "light and try again."
                ),
            )

        ear_open = float(np.median(self._ear_samples))
        mar_closed = float(np.median(self._mar_samples))

        raw_ear = ear_open * self.ear_ratio
        raw_mar = mar_closed + self.mar_offset
        ear_threshold = _clamp(
            raw_ear, config.EAR_THRESHOLD_MIN, config.EAR_THRESHOLD_MAX
        )
        mar_threshold = _clamp(
            raw_mar, config.MAR_THRESHOLD_MIN, config.MAR_THRESHOLD_MAX
        )
        clamped = (ear_threshold != raw_ear) or (mar_threshold != raw_mar)

        message = (
            f"Calibrated: EAR threshold {ear_threshold:.2f} "
            f"(open eyes read {ear_open:.2f}), MAR threshold "
            f"{mar_threshold:.2f} (closed mouth reads {mar_closed:.2f})."
        )
        if clamped:
            message += (
                " One value was limited to the allowed range, so check that "
                "your eyes were open and your mouth was closed."
            )

        return CalibrationResult(
            success=True,
            ear_threshold=ear_threshold,
            mar_threshold=mar_threshold,
            ear_open=ear_open,
            mar_closed=mar_closed,
            samples=n,
            clamped=clamped,
            message=message,
        )