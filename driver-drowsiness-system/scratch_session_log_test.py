import csv
import tempfile

from src import config
from src.analytics import AnalyticsResult
from src.calibration import CalibrationResult
from src.session_log import SessionLogger

A = config.STATE_ALERT
M = config.STATE_MILD_FATIGUE
C = config.STATE_CRITICAL_DROWSY


def res(state, face=True, perclos=0.0, yawns=0):
    return AnalyticsResult(
        state=state, perclos=perclos, eyes_closed=False, yawning=False,
        closed_streak=0, yawn_count=yawns, face_detected=face,
    )


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


with tempfile.TemporaryDirectory() as tmp:
    # 1. The file starts with a header and a SESSION_START row
    log = SessionLogger(log_dir=tmp, now=0.0)
    r = rows(log.path)
    assert list(r[0].keys()) == config.LOG_COLUMNS
    assert [x["event"] for x in r] == ["SESSION_START"]
    assert r[0]["state"] == A

    # 2. State changes are logged and counted; time per state adds up
    for t, s in enumerate([A, A, M, M, C, A], start=1):
        log.update(res(s), 0.30, 0.20, now=float(t))
    changes = [x for x in rows(log.path) if x["event"] == "STATE_CHANGE"]
    assert [x["state"] for x in changes] == [M, C, A]
    s = log.summary(now=6.0)
    assert s["warnings"] == 1 and s["critical_alerts"] == 1
    assert abs(s["seconds_alert"] - 3) < 1e-9
    assert abs(s["seconds_mild"] - 2) < 1e-9
    assert abs(s["seconds_critical"] - 1) < 1e-9
    assert abs(s["duration_sec"] - 6.0) < 1e-9

    # 3. Escalation counting: jumping to critical, dropping to mild, and
    #    going critical again gives 2 critical alerts and no warnings
    log2 = SessionLogger(log_dir=tmp, now=0.0)
    for t, st in enumerate([C, M, C], start=1):
        log2.update(res(st), 0.10, 0.20, now=float(t))
    s2 = log2.summary(now=3.0)
    assert s2["critical_alerts"] == 2 and s2["warnings"] == 0
    assert log2.path != log.path            # same-second sessions get unique files

    # 4. Yawns: only increases count, even if the analytics count resets
    log3 = SessionLogger(log_dir=tmp, now=0.0)
    for t, y in enumerate([0, 1, 1, 2, 0, 1], start=1):
        log3.update(res(A, yawns=y), 0.30, 0.70, now=float(t))
    yawn_rows = [x for x in rows(log3.path) if x["event"] == "YAWN"]
    assert [x["yawn_count"] for x in yawn_rows] == ["1", "2", "3"]
    assert log3.summary(now=6.0)["yawns"] == 3

    # 5. Frames with no face never change the state or write rows
    log4 = SessionLogger(log_dir=tmp, now=0.0)
    log4.update(res(A, face=False), None, None, now=1.0)
    log4.update(res(A, face=False), None, None, now=2.0)
    assert [x["event"] for x in rows(log4.path)] == ["SESSION_START"]
    assert abs(log4.summary(now=2.0)["seconds_no_face"] - 2.0) < 1e-9

    # 6. Peak PERCLOS is tracked
    log5 = SessionLogger(log_dir=tmp, now=0.0)
    for t, p in enumerate([0.10, 0.40, 0.20], start=1):
        log5.update(res(A, perclos=p), 0.30, 0.20, now=float(t))
    assert abs(log5.summary(now=3.0)["peak_perclos"] - 0.40) < 1e-9

    # 7. Calibration events record the thresholds in use
    log6 = SessionLogger(log_dir=tmp, now=0.0)
    ok = CalibrationResult(True, 0.225, 0.45, 0.30, 0.15, 150, False, "ok")
    bad = CalibrationResult(False, 0.22, 0.50, 0.0, 0.0, 5, False, "bad")
    log6.log_calibration(ok)
    log6.log_calibration(bad)
    r6 = rows(log6.path)
    assert [x["event"] for x in r6[-2:]] == ["CALIBRATION_OK", "CALIBRATION_FAILED"]
    assert abs(float(r6[-2]["ear"]) - 0.225) < 1e-9
    assert abs(float(r6[-2]["mar"]) - 0.45) < 1e-9

    # 8. Closing writes SESSION_END, is safe to repeat, and blocks later rows
    log.close(now=7.0)
    log.close(now=8.0)
    assert rows(log.path)[-1]["event"] == "SESSION_END"
    assert abs(log.summary()["duration_sec"] - 7.0) < 1e-9
    count = len(rows(log.path))
    log.update(res(C), 0.10, 0.20, now=9.0)
    assert len(rows(log.path)) == count
    assert log.read_bytes().startswith(b"timestamp,event,state")
    assert "Critical alerts: 1" in log.summary_text()

    # Close every logger before the temp folder is deleted (needed on Windows)
    for lg in (log2, log3, log4, log5, log6):
        lg.close()

print("All session log tests passed")