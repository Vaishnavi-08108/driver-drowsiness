import io
import json
import tempfile
import urllib.error
from pathlib import Path

from src import config
from src.contacts import ContactBook
from src.emergency import (
    EVENT_ALERT_DONE, EVENT_ALERT_SENDING, EVENT_COUNTDOWN_RECOVERED,
    EVENT_COUNTDOWN_STARTED, PHASE_COOLDOWN, PHASE_COUNTDOWN, PHASE_IDLE,
    PHASE_SENDING, RECOVERY_SECONDS, EmergencyManager, SendResult,
    TelegramClient, build_alert_text,
)
from src.location import SOURCE_BROWSER, SOURCE_MANUAL, LocationProvider, make_location

CRIT, ALERT = config.STATE_CRITICAL_DROWSY, config.STATE_ALERT
TOKEN = "123456:SECRET-TOKEN"
PLACE = make_location(12.9716, 77.5946, SOURCE_MANUAL, now=1.0)


# ---------------------------------------------------------------- fakes
class FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self, n=-1):
        return self._body[:n] if n > 0 else self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeClient:
    configured = True

    def __init__(self, fail_for=()):
        self.messages, self.photos, self.fail_for = [], [], set(fail_for)

    def send_message(self, chat_id, text):
        self.messages.append((chat_id, text))
        if chat_id in self.fail_for:
            return SendResult(False, "HTTP 403: Forbidden: bot was blocked by the user")
        return SendResult(True)

    def send_photo(self, chat_id, jpeg, caption=""):
        self.photos.append((chat_id, jpeg))
        return SendResult(True)


class FakeLocations:
    def current(self):
        return PLACE


def feed(mgr, state, t0, t1, face=True, step=0.5):
    """Feed frames from t0 up to and including t1."""
    status, t = None, t0
    while t <= t1 + 1e-9:
        status = mgr.update(state, face, now=t)
        t += step
    return status


_mgr_seq = 0


def make_manager(tmp, client=None, dry_run=False, n_contacts=2, **kw):
    global _mgr_seq
    _mgr_seq += 1
    book = ContactBook(Path(tmp) / f"contacts_{_mgr_seq}_{n_contacts}.json")
    for i in range(n_contacts):
        book.add(f"Contact {i}", f"90000000{i}")
    return EmergencyManager(
        book, FakeLocations(), snapshot_provider=lambda: b"\xff\xd8JPEG",
        client=client or FakeClient(), driver_name="Vaishnavi", dry_run=dry_run,
        threaded=False, retry_delay=0, wall_clock=lambda: 1_700_000_000.0, **kw,
    )


with tempfile.TemporaryDirectory() as tmp:
    # 1. Critical for under 10 s does nothing; at 10 s the countdown starts
    m = make_manager(tmp)
    s = feed(m, CRIT, 0, 9.5)
    assert s.phase == PHASE_IDLE and abs(s.critical_seconds - 9.5) < 1e-9
    s = m.update(CRIT, True, now=10.0)
    assert s.phase == PHASE_COUNTDOWN and s.event == EVENT_COUNTDOWN_STARTED
    assert abs(s.countdown_remaining - 15.0) < 1e-9

    # 2. Any break in the critical state restarts the 10 s timer
    m = make_manager(tmp)
    feed(m, CRIT, 0, 6)
    m.update(ALERT, True, now=6.5)
    s = feed(m, CRIT, 7, 14.5)                     # only 7.5 s since the restart
    assert s.phase == PHASE_IDLE

    # 3. Cancel stops the countdown, and the timer starts again from zero
    m = make_manager(tmp)
    feed(m, CRIT, 0, 10)
    assert m.cancel() is True and m.cancel() is False
    s = m.update(CRIT, True, now=10.5)
    assert s.phase == PHASE_IDLE and s.critical_seconds == 0.0

    # 4. Being seen awake for 3 s cancels it; a shorter spell does not
    m = make_manager(tmp)
    feed(m, CRIT, 0, 10)
    s = feed(m, ALERT, 10.5, 10.5 + RECOVERY_SECONDS)
    assert s.phase == PHASE_IDLE and s.event == EVENT_COUNTDOWN_RECOVERED
    m = make_manager(tmp)
    feed(m, CRIT, 0, 10)
    feed(m, ALERT, 10.5, 12.0)                     # awake for 1.5 s only
    s = m.update(CRIT, True, now=12.5)             # critical again resets that timer
    s = feed(m, ALERT, 13, 15)                     # awake for 2 s only
    assert s.phase == PHASE_COUNTDOWN

    # 5. No face never cancels the countdown, and never starts one
    m = make_manager(tmp)
    s = feed(m, CRIT, 0, 30, face=False)
    assert s.phase == PHASE_IDLE
    m = make_manager(tmp)
    feed(m, CRIT, 0, 10)
    s = feed(m, ALERT, 10.5, 20, face=False)       # driver slumps out of view
    assert s.phase == PHASE_COUNTDOWN
    s = feed(m, ALERT, 20.5, 30, face=False)
    assert s.phase in (PHASE_SENDING, PHASE_COOLDOWN)

    # 6. Expiry sends to every contact with location, snapshot, and no token leak
    client = FakeClient()
    m = make_manager(tmp, client=client)
    feed(m, CRIT, 0, 10)
    s = feed(m, CRIT, 10.5, 25)
    assert s.event == EVENT_ALERT_SENDING and s.phase == PHASE_SENDING
    assert len(client.messages) == 2 and len(client.photos) == 2
    text = client.messages[0][1]
    assert "Vaishnavi" in text and "maps?q=12.97160,77.59460" in text
    s = m.update(CRIT, True, now=25.5)
    assert s.phase == PHASE_COOLDOWN and s.event == EVENT_ALERT_DONE
    assert s.report.delivered == 2 and "delivered to 2 of 2" in s.report.summary()
    assert "******" in s.report.recipients[0].chat_id          # masked in the report

    # 6b. Browser GPS updates on the shared provider reach real alert messages.
    browser_locations = LocationProvider(ip_fetcher=lambda now=None: None)
    assert browser_locations.update_from_browser(
        {"latitude": 12.9716, "longitude": 77.5946, "accuracy": 18.0}
    )
    browser_book = ContactBook(Path(tmp) / "browser_contacts.json")
    browser_book.add("Contact", "900000001")
    browser_client = FakeClient()
    browser_manager = EmergencyManager(
        browser_book,
        browser_locations,
        snapshot_provider=lambda: b"\xff\xd8JPEG",
        client=browser_client,
        driver_name="Vaishnavi",
        dry_run=False,
        threaded=False,
        retry_delay=0,
        wall_clock=lambda: 1_700_000_000.0,
    )
    feed(browser_manager, CRIT, 0, 25.5)
    assert len(browser_client.messages) == 1
    assert "maps?q=12.97160,77.59460" in browser_client.messages[0][1]
    assert browser_manager.report.location.source == SOURCE_BROWSER

    # 7. The cooldown blocks a second alert until it ends, then it works again
    client = FakeClient()
    m = make_manager(tmp, client=client, cooldown_seconds=300)
    feed(m, CRIT, 0, 25)
    assert len(client.messages) == 2
    assert m.update(CRIT, True, now=25.5).phase == PHASE_COOLDOWN
    s = feed(m, CRIT, 26, 120)
    assert s.phase == PHASE_COOLDOWN and len(client.messages) == 2
    s = m.update(CRIT, True, now=10 + 15 + 300)
    assert s.phase == PHASE_SENDING and s.event == EVENT_ALERT_SENDING
    assert len(client.messages) == 4

    # 8. Dry run builds the report and sends nothing
    client = FakeClient()
    m = make_manager(tmp, client=client, dry_run=True)
    feed(m, CRIT, 0, 26)
    s = m.update(CRIT, True, now=26.5)
    assert client.messages == [] and client.photos == []
    assert s.report.dry_run and len(s.report.recipients) == 2
    assert "nothing was sent" in s.report.summary()

    # 9. Not ready (live mode, no contacts): no countdown, with a reason
    m = make_manager(tmp, n_contacts=0)
    s = feed(m, CRIT, 0, 30)
    assert s.phase == PHASE_IDLE and not s.ready and "contact" in s.problems[0]
    unconfigured = TelegramClient(token="")
    m = EmergencyManager(ContactBook(Path(tmp) / "x.json"), client=unconfigured, dry_run=False)
    ready, problems = m.readiness()
    assert not ready and any("token" in p for p in problems)
    m.dry_run = True
    assert m.readiness()[0] is True

    # 10. A failed contact is reported while the others still get the alert
    client = FakeClient(fail_for={"900000000"})
    m = make_manager(tmp, client=client)
    feed(m, CRIT, 0, 26)
    r = m.update(CRIT, True, now=26.5).report
    assert r.delivered == 1 and "blocked" in r.summary()

    # 11. A snapshot that fails does not stop the alert
    def broken_snapshot():
        raise RuntimeError("camera gone")
    m = make_manager(tmp)
    m._snapshot_provider = broken_snapshot
    feed(m, CRIT, 0, 26)
    r = m.update(CRIT, True, now=26.5).report
    assert r.delivered == 2 and r.photo_attached is False

    # 12. Threaded sending produces a report once joined
    m = make_manager(tmp)
    m.threaded = True
    feed(m, CRIT, 0, 25.5)
    m.join()
    s = m.update(CRIT, True, now=26.0)
    assert s.phase == PHASE_COOLDOWN and s.report.delivered == 2

    # 13. Reset clears the cycle
    m.reset()
    assert m.phase == PHASE_IDLE and m.report is None

# 14. Message text
text = build_alert_text("  Asha\nKumar  ", 1_700_000_000.0, None, 15)
assert "Asha Kumar may have fallen asleep" in text       # newline removed
assert "Location: not available" in text and "15 second" in text
assert build_alert_text("", 1_700_000_000.0, PLACE, 15).splitlines()[1].startswith("The driver")


# ---------------------------------------------------------------- Telegram client
def recording_opener(reply, store):
    def opener(request, timeout=None):
        store.append(request)
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply if isinstance(reply, bytes) else json.dumps(reply).encode())
    return opener


sent = []
ok_client = TelegramClient(TOKEN, opener=recording_opener({"ok": True, "result": {}}, sent))

# 15. sendMessage goes to the right URL with a numeric chat id
assert ok_client.send_message("123456789", "hello").ok
assert sent[0].full_url.endswith(f"/bot{TOKEN}/sendMessage")
body = json.loads(sent[0].data)
assert body["chat_id"] == 123456789 and body["text"] == "hello"

# 16. sendPhoto builds a multipart body that contains the image bytes
sent.clear()
assert ok_client.send_photo("-1001234567890", b"\xff\xd8IMG", "caption").ok
req = sent[0]
assert req.get_header("Content-type").startswith("multipart/form-data; boundary=")
assert b'name="photo"; filename="snapshot.jpg"' in req.data and b"\xff\xd8IMG" in req.data
assert b"-1001234567890" in req.data

# 17. Errors never contain the token
blocked = urllib.error.HTTPError(
    "u", 403, "Forbidden", {}, io.BytesIO(
        json.dumps({"ok": False, "description": f"Forbidden: bot was blocked {TOKEN}"}).encode()))
r = TelegramClient(TOKEN, opener=recording_opener(blocked, [])).send_message("123456789", "x")
assert not r.ok and "403" in r.error and "blocked" in r.error and TOKEN not in r.error
r = TelegramClient(TOKEN, opener=recording_opener(urllib.error.URLError(f"fail {TOKEN}"), [])).send_message("123456789", "x")
assert not r.ok and TOKEN not in r.error
r = TelegramClient(TOKEN, opener=recording_opener(b"not json", [])).send_message("123456789", "x")
assert not r.ok
r = TelegramClient(TOKEN, opener=recording_opener({"ok": False, "description": "chat not found"}, [])).send_message("123456789", "x")
assert not r.ok and "chat not found" in r.error

# 18. No token means no network call at all
calls = []
no_token = TelegramClient("", opener=recording_opener({"ok": True}, calls))
assert not no_token.configured and not no_token.send_message("123456789", "x").ok and calls == []
assert not TelegramClient(TOKEN, base_url="http://insecure.example").configured

# 19. Chat discovery de-duplicates and names chats
updates = {"ok": True, "result": [
    {"message": {"chat": {"id": 111111111, "type": "private", "first_name": "Asha", "last_name": "K"}}},
    {"message": {"chat": {"id": 111111111, "type": "private", "first_name": "Asha"}}},
    {"message": {"chat": {"id": -100222222222, "type": "group", "title": "Family"}}},
    {"edited_message": {"chat": {"id": 5}}},
    {"message": {"chat": {"type": "private"}}},
]}
chats, err = TelegramClient(TOKEN, opener=recording_opener(updates, [])).get_chat_ids()
assert err == "" and [(c.chat_id, c.name) for c in chats] == [("111111111", "Asha K"), ("-100222222222", "Family")]
chats, err = TelegramClient(TOKEN, opener=recording_opener(urllib.error.URLError("down"), [])).get_chat_ids()
assert chats == [] and err

# 20. Retry: a network fault is retried once, a 4xx reply is not
class Flaky:
    configured = True
    def __init__(self, results):
        self.results, self.n = list(results), 0
    def send_message(self, chat_id, text):
        self.n += 1
        return self.results.pop(0)
    def send_photo(self, *a, **k):
        return SendResult(True)

with tempfile.TemporaryDirectory() as tmp:
    flaky = Flaky([SendResult(False, "Network problem: timed out"), SendResult(True)])
    m = make_manager(tmp, client=flaky, n_contacts=1)
    feed(m, CRIT, 0, 26)
    assert flaky.n == 2 and m.update(CRIT, True, now=26.5).report.delivered == 1
    firm = Flaky([SendResult(False, "HTTP 400: chat not found")])
    m = make_manager(tmp, client=firm, n_contacts=1)
    feed(m, CRIT, 0, 26)
    assert firm.n == 1

# 21. A continuous critical episode sends a fresh snapshot after cooldown
with tempfile.TemporaryDirectory() as tmp:
    client = FakeClient()
    m = make_manager(tmp, client=client, n_contacts=1, cooldown_seconds=60)
    snapshots = []
    m._snapshot_provider = lambda: snapshots.append(len(snapshots) + 1) or f"snapshot-{len(snapshots)}".encode()

    feed(m, CRIT, 0, 25)
    assert len(client.messages) == 1 and len(client.photos) == 1
    assert client.photos[0][1] == b"snapshot-1"
    assert m.update(CRIT, True, now=25.5).phase == PHASE_COOLDOWN
    feed(m, CRIT, 26, 84.5)
    assert len(client.messages) == 1

    s = m.update(CRIT, True, now=85)
    assert s.phase == PHASE_SENDING and s.event == EVENT_ALERT_SENDING
    assert len(client.messages) == 2 and len(client.photos) == 2
    assert client.photos[1][1] == b"snapshot-2"

# 22. Recovery during cooldown prevents an immediate repeat; a new episode
#     must pass the original critical and countdown timers.
with tempfile.TemporaryDirectory() as tmp:
    client = FakeClient()
    m = make_manager(tmp, client=client, n_contacts=1, cooldown_seconds=60)
    feed(m, CRIT, 0, 25)
    m.update(CRIT, True, now=25.5)
    m.update(ALERT, True, now=30)

    s = m.update(ALERT, True, now=85)
    assert s.phase == PHASE_IDLE and len(client.messages) == 1

    m.update(CRIT, True, now=85.5)
    s = m.update(CRIT, True, now=95.5)
    assert s.phase == PHASE_COUNTDOWN and len(client.messages) == 1
    s = m.update(CRIT, True, now=110.5)
    assert s.phase == PHASE_SENDING and len(client.messages) == 2

print("All emergency tests passed")