from src import config
from src.analytics import DrowsinessAnalytics

OPEN, CLOSED = 0.30, 0.10      # EAR values
MOUTH_SHUT, MOUTH_OPEN = 0.20, 0.70


def feed(analytics, ear, mar, frames):
    result = None
    for _ in range(frames):
        result = analytics.update(ear, mar)
    return result


# 1. Steady open eyes stay ALERT
a = DrowsinessAnalytics()
assert feed(a, OPEN, MOUTH_SHUT, 100).state == config.STATE_ALERT

# 2. Eyes shut for 20 frames (more than 15) go CRITICAL
a = DrowsinessAnalytics()
feed(a, OPEN, MOUTH_SHUT, 100)
assert feed(a, CLOSED, MOUTH_SHUT, 20).state == config.STATE_CRITICAL_DROWSY

# 3. A 5-frame blink does not trigger anything
a = DrowsinessAnalytics()
feed(a, OPEN, MOUTH_SHUT, 100)
feed(a, CLOSED, MOUTH_SHUT, 5)
assert feed(a, OPEN, MOUTH_SHUT, 20).state == config.STATE_ALERT

# 4. A 20-frame yawn gives MILD_FATIGUE and counts one yawn
a = DrowsinessAnalytics()
feed(a, OPEN, MOUTH_SHUT, 50)
r = feed(a, OPEN, MOUTH_OPEN, 20)
assert r.state == config.STATE_MILD_FATIGUE and r.yawn_count == 1

# 5. Frequent short closures (about 20 percent) raise PERCLOS to MILD
a = DrowsinessAnalytics()
for i in range(300):
    r = a.update(CLOSED if i % 30 < 6 else OPEN, MOUTH_SHUT)
assert 0.15 <= r.perclos < 0.30, r.perclos
assert r.state == config.STATE_MILD_FATIGUE

# 6. Missing face frames are ignored, not counted as closed eyes
a = DrowsinessAnalytics()
feed(a, OPEN, MOUTH_SHUT, 100)
r = feed(a, None, None, 50)
assert r.face_detected is False and r.perclos == 0.0

# 7. Warm-up: one closed frame at startup does not read as 100 percent
a = DrowsinessAnalytics()
assert a.update(CLOSED, MOUTH_SHUT).perclos == 0.0

print("All analytics tests passed")