"""
emergency.py
Escalation logic and Telegram delivery for the Driver Drowsiness Detection System.

How an alert works:

    1. The driver stays in CRITICAL_DROWSY for EMERGENCY_CRITICAL_SECONDS
       without a break.
    2. A countdown of EMERGENCY_COUNTDOWN_SECONDS starts. The driver can stop
       it with the "I'm awake" button (cancel), or by being seen awake for
       RECOVERY_SECONDS in a row.
    3. If the countdown runs out, a Telegram message (and a snapshot) goes to
       every saved contact. The sending runs in a background thread, so the
       video never freezes while the network is slow.
    4. After an alert, nothing new is sent for EMERGENCY_COOLDOWN_SECONDS.

Dry run mode builds the same message but sends nothing. Leave it on while
testing.

Safety choices:
    * A frame with no face never cancels a countdown, because a driver who
      slumps out of view has not recovered.
    * A frame with no face never starts a countdown either, because an empty
      seat is not an emergency.
    * The bot token is read from an environment variable, and it is removed
      from every error message before the message is stored or shown.
    * Message text is plain text (no Markdown), so a contact name cannot
      change how the message is displayed.

All time-dependent methods accept a `now` argument, so the test script can run
a whole timeline without waiting.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple

from src import config
from src.contacts import mask_chat_id
from src.location import Location, describe, map_link, valid_coordinates, make_location

PHASE_IDLE = "IDLE"
PHASE_COUNTDOWN = "COUNTDOWN"
PHASE_SENDING = "SENDING"
PHASE_COOLDOWN = "COOLDOWN"

# Event names, reported on the frame where they happen so the app can log them
EVENT_COUNTDOWN_STARTED = "EMERGENCY_COUNTDOWN_STARTED"
EVENT_COUNTDOWN_RECOVERED = "EMERGENCY_COUNTDOWN_RECOVERED"
EVENT_ALERT_SENDING = "EMERGENCY_ALERT_SENDING"
EVENT_ALERT_DONE = "EMERGENCY_ALERT_DONE"

# During a countdown, the driver must be seen awake this long in a row to
# cancel it automatically.
RECOVERY_SECONDS = 3.0

MAX_RESPONSE_BYTES = 65536
MAX_NAME_LENGTH = 50
PHOTO_CAPTION = "Snapshot taken when the alert was triggered."


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SendResult:
    ok: bool
    error: str = ""


@dataclass(frozen=True)
class ChatInfo:
    """A chat that has talked to the bot, found through getUpdates."""

    chat_id: str
    name: str
    kind: str      # "private", "group", ...


class TelegramClient:
    """Minimal Telegram Bot API client built on the standard library."""

    def __init__(
        self,
        token: Optional[str] = None,
        opener: Callable[..., Any] = urllib.request.urlopen,
        base_url: str = config.TELEGRAM_API_BASE,
        timeout: float = config.TELEGRAM_TIMEOUT_SEC,
    ) -> None:
        raw = token if token is not None else os.environ.get(config.TELEGRAM_TOKEN_ENV, "")
        self._token = str(raw).strip()
        self._opener = opener
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self._token) and self._base_url.lower().startswith("https://")

    # -----------------------------------------------------------------------
    # Public calls
    # -----------------------------------------------------------------------
    def send_message(self, chat_id: str, text: str) -> SendResult:
        body = json.dumps({"chat_id": int(chat_id), "text": text[:4096]}).encode("utf-8")
        ok, _, error = self._call("sendMessage", body, "application/json")
        return SendResult(ok, error)

    def phone_location(self) -> Optional[Location]:
        """Read recent live-location updates from the explicitly paired driver chat."""
        driver_chat = os.environ.get("DRIVER_LOCATION_CHAT_ID", "").strip()
        if not driver_chat:
            return None
        body = json.dumps({"limit": 100, "timeout": 0,
                           "allowed_updates": ["message", "edited_message"]}).encode("utf-8")
        ok, updates, _ = self._call("getUpdates", body, "application/json")
        if not ok or not isinstance(updates, list):
            return None
        latest = None
        for update in updates:
            msg = update.get("edited_message") or update.get("message") or {}
            chat = msg.get("chat", {})
            if chat.get("type") != "private" or str(chat.get("id")) != driver_chat:
                continue
            loc = msg.get("location")
            if not isinstance(loc, dict):
                continue
            stamp = msg.get("edit_date", msg.get("date", 0))
            if latest is None or stamp >= latest[0]:
                latest = (stamp, loc, msg)
        if latest is None:
            return None
        stamp, loc, msg = latest
        now = time.time()
        period = loc.get("live_period", 0)
        if not period or now >= msg.get("date", 0) + period or not 0 <= now - stamp <= 60:
            return None
        lat, lon = loc.get("latitude"), loc.get("longitude")
        if not valid_coordinates(lat, lon):
            return None
        fix = make_location(lat, lon, "phone_telegram", accuracy_m=loc.get("horizontal_accuracy"), now=stamp)
        if fix.accuracy_m is not None and fix.accuracy_m > 50:
            return None
        return fix

    def send_location(self, chat_id: str, location: Location) -> SendResult:
        body = json.dumps({"chat_id": int(chat_id), "latitude": location.lat,
                           "longitude": location.lon}).encode("utf-8")
        ok, _, error = self._call("sendLocation", body, "application/json")
        return SendResult(ok, error)

    def send_photo(self, chat_id: str, jpeg: bytes, caption: str = "") -> SendResult:
        boundary = "----dds" + uuid.uuid4().hex
        parts = [
            self._field(boundary, "chat_id", str(int(chat_id))),
            self._field(boundary, "caption", caption[:1000]),
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="photo"; filename="snapshot.jpg"\r\n'
                "Content-Type: image/jpeg\r\n\r\n"
            ).encode("utf-8") + jpeg + b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        ok, _, error = self._call(
            "sendPhoto", b"".join(parts), f"multipart/form-data; boundary={boundary}"
        )
        return SendResult(ok, error)

    def get_chat_ids(self) -> Tuple[List[ChatInfo], str]:
        """
        List chats that have messaged the bot recently, so the user can pick
        the right chat ID. Telegram keeps updates for about 24 hours, and this
        call does not remove them.

        Returns:
            (chats, error). `error` is empty on success.
        """
        body = json.dumps({"limit": 100, "timeout": 0}).encode("utf-8")
        ok, result, error = self._call("getUpdates", body, "application/json")
        if not ok:
            return [], error

        chats: List[ChatInfo] = []
        seen = set()
        for update in result if isinstance(result, list) else []:
            message = update.get("message") if isinstance(update, dict) else None
            chat = message.get("chat") if isinstance(message, dict) else None
            if not isinstance(chat, dict) or "id" not in chat:
                continue
            chat_id = str(chat["id"])
            if chat_id in seen:
                continue
            seen.add(chat_id)
            name = (
                chat.get("title")
                or " ".join(p for p in (chat.get("first_name"), chat.get("last_name")) if p)
                or chat.get("username")
                or "Unknown"
            )
            chats.append(ChatInfo(chat_id, str(name)[:MAX_NAME_LENGTH], str(chat.get("type", ""))))
        return chats, ""

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    @staticmethod
    def _field(boundary: str, name: str, value: str) -> bytes:
        return (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode("utf-8")

    def _clean(self, text: str) -> str:
        """Remove the token from any text and keep it short."""
        if self._token:
            text = text.replace(self._token, "[token]")
        return text[:200]

    def _call(self, method: str, body: bytes, content_type: str) -> Tuple[bool, Any, str]:
        """Returns (ok, result, error). Never raises."""
        if not self.configured:
            return False, None, "The Telegram bot token is not set."

        request = urllib.request.Request(
            f"{self._base_url}/bot{self._token}/{method}",
            data=body,
            headers={
                "Content-Type": content_type,
                "User-Agent": "driver-drowsiness-system/1.0",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                data = json.loads(response.read(MAX_RESPONSE_BYTES).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return False, None, self._clean(f"HTTP {exc.code}: {self._error_description(exc)}")
        except (OSError, ValueError) as exc:
            return False, None, self._clean(f"Network problem: {exc}")

        if not isinstance(data, dict) or not data.get("ok"):
            description = data.get("description") if isinstance(data, dict) else None
            return False, None, self._clean(f"Telegram refused the request: {description or 'unexpected reply'}")
        return True, data.get("result"), ""

    @staticmethod
    def _error_description(exc: urllib.error.HTTPError) -> str:
        try:
            data = json.loads(exc.read(MAX_RESPONSE_BYTES).decode("utf-8"))
            return str(data.get("description") or exc.reason)
        except (OSError, ValueError, AttributeError):
            return str(exc.reason)


def _clean_name(name: str) -> str:
    return " ".join(str(name).split())[:MAX_NAME_LENGTH]


def build_alert_text(
    driver_name: str,
    wall_time: float,
    location: Optional[Location],
    countdown_seconds: float,
) -> str:
    """The plain-text message sent to the contacts."""
    who = _clean_name(driver_name) or "The driver"
    when = datetime.fromtimestamp(wall_time).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "EMERGENCY ALERT",
        f"{who} may have fallen asleep while driving.",
        f"Time: {when}",
    ]
    if location is not None:
        lines.append(f"Location: {describe(location)}")
        if location.source == "phone_telegram":
            lines.append("Source: driver phone live location; accuracy not reported." if location.accuracy_m is None else "Source: driver phone live location.")
        lines.append(map_link(location))
    else:
        lines.append("Location: no recent accurate device fix available; please confirm the driver’s location directly.")
    lines.append(
        f"There was no response to the {int(countdown_seconds)} second check. "
        "Please try to call them now."
    )
    lines.append("Sent automatically by the Driver Drowsiness Detection System.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reports and status
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecipientResult:
    name: str
    chat_id: str           # masked
    message_ok: bool
    photo_ok: bool
    error: str = ""


@dataclass(frozen=True)
class AlertReport:
    wall_time: float
    dry_run: bool
    text: str
    location: Optional[Location]
    photo_attached: bool
    recipients: Tuple[RecipientResult, ...]
    error: str = ""

    @property
    def delivered(self) -> int:
        return sum(1 for r in self.recipients if r.message_ok)

    def summary(self) -> str:
        if self.error:
            return f"The alert failed: {self.error}"
        count = len(self.recipients)
        if self.dry_run:
            return f"Dry run: the alert was prepared for {count} contact(s) and nothing was sent."
        if count == 0:
            return "There were no contacts to notify."
        lines = [f"Alert delivered to {self.delivered} of {count} contact(s)."]
        lines += [f"{r.name}: {r.error}" for r in self.recipients if r.error]
        return "\n".join(lines)


@dataclass(frozen=True)
class EmergencyStatus:
    """What the dashboard needs after each frame."""

    phase: str
    critical_seconds: float          # time spent critical so far (IDLE only)
    countdown_remaining: float       # seconds left to press "I'm awake"
    cooldown_remaining: float
    ready: bool                      # False means an alert could not be sent
    problems: Tuple[str, ...]        # why it is not ready
    event: str                       # one of the EVENT_* values, or ""
    report: Optional[AlertReport]    # the most recent alert


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class EmergencyManager:
    """Runs the critical, countdown, send, cooldown cycle."""

    def __init__(
        self,
        contact_book: Any,
        location_provider: Any = None,
        snapshot_provider: Optional[Callable[[], Optional[bytes]]] = None,
        client: Optional[TelegramClient] = None,
        driver_name: str = "",
        dry_run: bool = config.EMERGENCY_DRY_RUN_DEFAULT,
        critical_seconds: float = config.EMERGENCY_CRITICAL_SECONDS,
        countdown_seconds: float = config.EMERGENCY_COUNTDOWN_SECONDS,
        cooldown_seconds: float = config.EMERGENCY_COOLDOWN_SECONDS,
        threaded: bool = True,
        retry_delay: float = 1.0,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._contacts = contact_book
        self._locations = location_provider
        self._snapshot_provider = snapshot_provider
        self._client = client if client is not None else TelegramClient()
        self.driver_name = driver_name
        self.dry_run = dry_run
        self.critical_seconds = critical_seconds
        self.countdown_seconds = countdown_seconds
        self.cooldown_seconds = cooldown_seconds
        self.threaded = threaded
        self.retry_delay = retry_delay
        self._wall_clock = wall_clock

        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self.reset()

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------
    def reset(self) -> None:
        """Forget the current cycle. A running send is allowed to finish."""
        with self._lock:
            self.phase = PHASE_IDLE
            self._critical_since: Optional[float] = None
            self._countdown_started = 0.0
            self._recovered_since: Optional[float] = None
            self._cooldown_until = 0.0
            self._repeat_eligible = False
            self._send_started = 0.0
            self._send_done = False
            self.report: Optional[AlertReport] = None

    def cancel(self) -> bool:
        """The "I'm awake" button. Returns True if a countdown was stopped."""
        with self._lock:
            if self.phase != PHASE_COUNTDOWN:
                return False
            self.phase = PHASE_IDLE
            self._critical_since = None
            self._recovered_since = None
            return True

    def readiness(self) -> Tuple[bool, Tuple[str, ...]]:
        """Can an alert actually be sent right now?"""
        problems: List[str] = []
        if not self.dry_run:
            if not self._client.configured:
                problems.append("The Telegram bot token is not set.")
            if len(self._contacts) == 0:
                problems.append("Add at least one emergency contact.")
        return (not problems, tuple(problems))

    def join(self, timeout: float = 5.0) -> None:
        """Wait for a background send to finish (used by tests and on exit)."""
        worker = self._worker
        if worker is not None:
            worker.join(timeout)

    # -----------------------------------------------------------------------
    # Per-frame update
    # -----------------------------------------------------------------------
    def update(
        self, state: str, face_detected: bool, now: Optional[float] = None
    ) -> EmergencyStatus:
        """Call once per frame with the analytics state."""
        now = time.monotonic() if now is None else now
        critical = face_detected and state == config.STATE_CRITICAL_DROWSY
        awake = face_detected and state != config.STATE_CRITICAL_DROWSY
        ready, problems = self.readiness()
        event = ""

        with self._lock:
            if self.phase in (PHASE_SENDING, PHASE_COOLDOWN):
                if critical:
                    if self._critical_since is None:
                        self._critical_since = now
                    self._recovered_since = None
                elif awake:
                    if self._recovered_since is None:
                        self._recovered_since = now
                    elif now - self._recovered_since >= RECOVERY_SECONDS:
                        self._critical_since = None
                        self._repeat_eligible = False
                else:
                    # Missing face is not evidence of recovery.
                    self._recovered_since = None

            # A finished background send moves us into the cooldown
            if self.phase == PHASE_SENDING and self._send_done:
                self.phase = PHASE_COOLDOWN
                self._cooldown_until = self._send_started + self.cooldown_seconds
                event = EVENT_ALERT_DONE

            if self.phase == PHASE_IDLE:
                event = self._update_idle(critical, ready, now) or event
            elif self.phase == PHASE_COUNTDOWN:
                event = self._update_countdown(critical, awake, now) or event
            elif self.phase == PHASE_COOLDOWN:
                if now >= self._cooldown_until:
                    if self._repeat_eligible and ready and not awake:
                        self._start_send(now, critical=critical)
                        event = EVENT_ALERT_SENDING
                    else:
                        self.phase = PHASE_IDLE
                        event = self._update_idle(critical, ready, now) or event

            return self._status(now, ready, problems, event)

    def _update_idle(self, critical: bool, ready: bool, now: float) -> str:
        if not critical:
            self._critical_since = None
            return ""
        if self._critical_since is None:
            self._critical_since = now
        if now - self._critical_since >= self.critical_seconds and ready:
            self.phase = PHASE_COUNTDOWN
            self._countdown_started = now
            self._recovered_since = None
            return EVENT_COUNTDOWN_STARTED
        return ""

    def _update_countdown(self, critical: bool, awake: bool, now: float) -> str:
        # Cancel automatically once the driver has been seen awake for a while
        if awake:
            if self._recovered_since is None:
                self._recovered_since = now
            elif now - self._recovered_since >= RECOVERY_SECONDS:
                self.phase = PHASE_IDLE
                self._critical_since = None
                self._recovered_since = None
                return EVENT_COUNTDOWN_RECOVERED
        else:
            # Still critical, or no face visible: not recovered
            self._recovered_since = None

        if now - self._countdown_started >= self.countdown_seconds:
            self._start_send(now, critical or not awake)
            return EVENT_ALERT_SENDING
        return ""

    def _status(
        self, now: float, ready: bool, problems: Tuple[str, ...], event: str
    ) -> EmergencyStatus:
        critical_for = 0.0
        if self.phase == PHASE_IDLE and self._critical_since is not None:
            critical_for = now - self._critical_since
        countdown_left = 0.0
        if self.phase == PHASE_COUNTDOWN:
            countdown_left = max(0.0, self.countdown_seconds - (now - self._countdown_started))
        cooldown_left = 0.0
        if self.phase == PHASE_COOLDOWN:
            cooldown_left = max(0.0, self._cooldown_until - now)
        return EmergencyStatus(
            phase=self.phase,
            critical_seconds=critical_for,
            countdown_remaining=countdown_left,
            cooldown_remaining=cooldown_left,
            ready=ready,
            problems=problems,
            event=event,
            report=self.report,
        )

    # -----------------------------------------------------------------------
    # Sending
    # -----------------------------------------------------------------------
    def _start_send(self, now: float, critical: bool) -> None:
        self.phase = PHASE_SENDING
        self._repeat_eligible = critical or self._repeat_eligible
        self._recovered_since = None
        self._send_started = now
        self._send_done = False

        # The snapshot is taken here, on the calling thread, where the
        # latest frame is available.
        jpeg: Optional[bytes] = None
        if self._snapshot_provider is not None:
            try:
                jpeg = self._snapshot_provider()
            except Exception:
                jpeg = None
        wall_time = self._wall_clock()

        if self.threaded:
            self._worker = threading.Thread(
                target=self._send_worker, args=(jpeg, wall_time), daemon=True
            )
            self._worker.start()
        else:
            self._send_worker(jpeg, wall_time)

    def _send_worker(self, jpeg: Optional[bytes], wall_time: float) -> None:
        try:
            report = self._deliver(jpeg, wall_time)
        except Exception as exc:  # a fault here must not vanish silently
            report = AlertReport(
                wall_time, self.dry_run, "", None, jpeg is not None, (),
                error=f"{type(exc).__name__}: {exc}"[:200],
            )
        with self._lock:
            self.report = report
            self._send_done = True

    def _deliver(self, jpeg: Optional[bytes], wall_time: float) -> AlertReport:
        location = None
        try:
            location = self._client.phone_location()
        except Exception:
            location = None

        text = build_alert_text(self.driver_name, wall_time, location, self.countdown_seconds)
        results = []
        for contact in self._contacts.contacts:
            masked = mask_chat_id(contact.chat_id)
            if self.dry_run:
                results.append(RecipientResult(contact.name, masked, True, jpeg is not None))
                continue

            message = self._with_retry(lambda c=contact: self._client.send_message(c.chat_id, text))
            photo_ok, error = False, message.error
            if location is not None:
                pin = self._client.send_location(contact.chat_id, location)
                if not pin.ok and not error:
                    error = f"Location pin failed: {pin.error}"
            if jpeg is not None:
                photo = self._client.send_photo(contact.chat_id, jpeg, PHOTO_CAPTION)
                photo_ok = photo.ok
                if not photo.ok and not error:
                    error = f"The message was sent, but the photo was not: {photo.error}"
            results.append(RecipientResult(contact.name, masked, message.ok, photo_ok, error))

        return AlertReport(
            wall_time, self.dry_run, text, location, jpeg is not None, tuple(results)
        )

    def _with_retry(self, send: Callable[[], SendResult]) -> SendResult:
        """Try twice. A 4xx reply (blocked bot, wrong chat) will not improve on a retry."""
        result = send()
        if not result.ok and not result.error.startswith("HTTP 4"):
            if self.retry_delay > 0:
                time.sleep(self.retry_delay)
            result = send()
        return result