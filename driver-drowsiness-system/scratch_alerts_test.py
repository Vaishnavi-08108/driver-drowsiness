import io
import wave

from src import config
from src.alerts import SoundAlert, generate_beep_wav

CRIT = config.STATE_CRITICAL_DROWSY
ALERT = config.STATE_ALERT


class FakePlaceholder:
    """Records what the alert would do to the page."""

    def __init__(self):
        self.calls = []

    def markdown(self, html, unsafe_allow_html=False):
        assert "<audio autoplay>" in html and unsafe_allow_html
        self.calls.append("play")

    def empty(self):
        self.calls.append("clear")


# 1. The generated WAV is valid: mono, 16-bit, the right length
wav = generate_beep_wav(880, 0.6, 22050)
with wave.open(io.BytesIO(wav)) as wf:
    assert wf.getnchannels() == 1
    assert wf.getsampwidth() == 2
    assert wf.getframerate() == 22050
    assert abs(wf.getnframes() / wf.getframerate() - 0.6) < 0.01

# Save it so you can listen to the beep yourself
with open("beep_test.wav", "wb") as f:
    f.write(wav)

# 2. Nothing plays while the driver is alert
ph = FakePlaceholder()
a = SoundAlert()
assert a.update(ALERT, ph, now=0.0) is False
assert ph.calls == []

# 3. Critical beeps at once, respects the cooldown, then beeps again
ph = FakePlaceholder()
a = SoundAlert(cooldown=2.0)
assert a.update(CRIT, ph, now=0.0) is True      # first beep
assert a.update(CRIT, ph, now=1.0) is False     # inside the cooldown
assert a.update(CRIT, ph, now=2.5) is True      # cooldown over
assert ph.calls == ["play", "clear", "play"]    # element removed between beeps
assert a.beep_count == 2

# 4. Recovering and then re-entering critical beeps immediately
a.update(ALERT, ph, now=3.5)                    # recovers
assert a.update(CRIT, ph, now=3.6) is True      # no waiting for the old timer

# 5. Disabled sound never plays
ph = FakePlaceholder()
a = SoundAlert(enabled=False)
assert a.update(CRIT, ph, now=0.0) is False
assert ph.calls == []

print("All alert tests passed. Open beep_test.wav to hear the tone.")