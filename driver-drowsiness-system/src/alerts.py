"""
alerts.py
Software sound alert for the Driver Drowsiness Detection System.

This replaces the hardware buzzer. SoundAlert builds a short beep in memory
and plays it in the browser through an HTML <audio> element, so it needs no
extra package and works on Windows, Linux, and macOS.

How the repeat logic works:
    * The beep plays as soon as the state becomes CRITICAL_DROWSY.
    * While the state stays critical, it repeats every SOUND_COOLDOWN_SEC.
    * When the state leaves critical, the next critical event beeps at once.

Browsers only allow autoplay after the user has interacted with the page.
Clicking the "Start monitoring" toggle counts as that interaction.
"""

import base64
import io
import time
import wave
from typing import Any, Optional

import numpy as np

from src import config

# The <audio> element is removed from the page this long after the beep
# ends, so the next beep creates a fresh element. A reused element would
# not autoplay again.
CLEANUP_MARGIN_SEC = 0.1


def generate_beep_wav(
    frequency: float = config.SOUND_FREQUENCY_HZ,
    duration: float = config.SOUND_DURATION_SEC,
    sample_rate: int = config.SOUND_SAMPLE_RATE,
) -> bytes:
    """
    Build a mono 16-bit WAV file containing a sine beep.

    A 10 ms fade in and fade out removes the click that a hard start or
    stop would make.

    Returns:
        The complete WAV file as bytes.
    """
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate
    tone = np.sin(2.0 * np.pi * frequency * t)

    fade = min(int(0.01 * sample_rate), n_samples // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]

    # Scale to 80 percent of full range to avoid clipping on cheap speakers
    samples = (tone * 0.8 * 32767).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)          # 16-bit samples
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return buffer.getvalue()


class SoundAlert:
    """Plays a repeating beep in the browser while the driver is critical."""

    def __init__(
        self,
        enabled: bool = True,
        cooldown: float = config.SOUND_COOLDOWN_SEC,
    ) -> None:
        """
        Args:
            enabled: master switch. When False, update() never plays.
            cooldown: minimum seconds between two beeps.
        """
        self.enabled = enabled
        self.cooldown = cooldown

        self.wav_bytes = generate_beep_wav()
        self._b64 = base64.b64encode(self.wav_bytes).decode("ascii")

        self.beep_count = 0                      # beeps played this session
        self._last_play: Optional[float] = None  # drives the cooldown
        self._shown_at: Optional[float] = None   # when the element appeared

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def update(
        self, state: str, placeholder: Any, now: Optional[float] = None
    ) -> bool:
        """
        Call once per frame with the current driver state.

        Args:
            state: one of the config.STATE_* values.
            placeholder: a Streamlit st.empty() slot (any object with
                .markdown() and .empty() methods also works, which is how
                the test script checks this class without a browser).
            now: current time in seconds. Leave as None to use the clock;
                tests pass fake times.

        Returns:
            True if a beep was started on this call.
        """
        now = time.monotonic() if now is None else now

        self._remove_finished_element(placeholder, now)

        if not self.enabled or state != config.STATE_CRITICAL_DROWSY:
            # Reset the cooldown so the next critical event beeps at once
            self._last_play = None
            return False

        if self._last_play is not None and now - self._last_play < self.cooldown:
            return False

        self._last_play = now
        self._shown_at = now
        self.beep_count += 1
        placeholder.markdown(self._audio_html(), unsafe_allow_html=True)
        return True

    def clear(self, placeholder: Any) -> None:
        """Remove the audio element immediately (used when monitoring stops)."""
        placeholder.empty()
        self._shown_at = None
        self._last_play = None

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    def _audio_html(self) -> str:
        return (
            "<audio autoplay>"
            f'<source src="data:audio/wav;base64,{self._b64}" type="audio/wav">'
            "</audio>"
        )

    def _remove_finished_element(self, placeholder: Any, now: float) -> None:
        """Delete the <audio> element once its beep has finished playing."""
        if self._shown_at is None:
            return
        if now - self._shown_at >= config.SOUND_DURATION_SEC + CLEANUP_MARGIN_SEC:
            placeholder.empty()
            self._shown_at = None