from src.calibration import Calibrator

FPS = 30


def run(ear_fn, mar_fn, seconds=6.0, calibrator=None):
    """Feed simulated frames at 30 FPS and return the finished Calibrator."""
    c = calibrator or Calibrator()
    c.start(now=0.0)
    for i in range(int(seconds * FPS)):
        t = i / FPS
        if c.add_sample(ear_fn(i), mar_fn(i), now=t):
            break
    return c


# 1. Normal run: open-eye EAR 0.30 and closed-mouth MAR 0.15
c = run(lambda i: 0.30, lambda i: 0.15)
r = c.result
assert r.success and not r.clamped
assert abs(r.ear_threshold - 0.225) < 1e-6      # 0.30 * 0.75
assert abs(r.mar_threshold - 0.45) < 1e-6       # 0.15 + 0.30
assert c.active is False

# 2. Blinks during calibration do not move the result (median is robust)
c = run(lambda i: 0.05 if i % 40 < 4 else 0.30, lambda i: 0.15)
assert abs(c.result.ear_open - 0.30) < 1e-6

# 3. No face for most of the run fails and keeps the defaults
c = run(lambda i: 0.30 if i < 10 else None,
        lambda i: 0.15 if i < 10 else None)
assert c.result.success is False
assert c.result.samples == 10
assert c.result.ear_threshold == 0.22 and c.result.mar_threshold == 0.50

# 4. Extreme values are clamped and flagged
c = run(lambda i: 0.60, lambda i: 0.90)         # 0.45 and 1.20 before clamping
r = c.result
assert r.success and r.clamped
assert r.ear_threshold == 0.40 and r.mar_threshold == 1.00

# 5. Progress goes from 0 to 1 and calibration ends at 5 seconds
c = Calibrator()
c.start(now=0.0)
assert c.progress(now=0.0) == 0.0
assert abs(c.progress(now=2.5) - 0.5) < 1e-6
assert c.add_sample(0.30, 0.15, now=4.9) is False
assert c.add_sample(0.30, 0.15, now=5.0) is True

# 6. Samples are ignored when no calibration is active
c = Calibrator()
assert c.add_sample(0.30, 0.15, now=0.0) is False
assert c.result is None

# 7. Starting again clears the old data
c = run(lambda i: 0.30, lambda i: 0.15)
c.start(now=100.0)
assert c.active and c.result is None

print("All calibration tests passed")