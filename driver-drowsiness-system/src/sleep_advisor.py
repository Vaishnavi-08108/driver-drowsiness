"""
sleep_advisor.py
Driver Sleep Logging and Travel Readiness Advisor for the Driver Drowsiness Detection System.

Features:
    * SleepLogBook: Persistent, atomic, crash-safe local storage for daily sleep logs (data/sleep_log.json).
    * ReadinessScorer: Deterministic "Is it safe to drive right now?" safety score (0-100),
      evaluating last night's deficit, multi-day sleep debt, continuous awake duration,
      and circadian risk windows (midnight to 6 AM, afternoon dip).
    * Actionable, rule-based recommendations tailored to the driver's risk factors.
    * AnthropicChatClient: Lightweight standard-library Anthropic Claude client for conversational
      sleep and fatigue consultation (optional; all core scoring and logging runs 100% offline).
"""

import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from src import config

# Verdict constants
VERDICT_SAFE = "SAFE"
VERDICT_CAUTION = "CAUTION"
VERDICT_UNSAFE = "UNSAFE"

# UI Verdict Badge Colors
VERDICT_COLORS = {
    VERDICT_SAFE: "#2ecc71",      # Green
    VERDICT_CAUTION: "#f1c40f",   # Yellow
    VERDICT_UNSAFE: "#e74c3c",    # Red
}

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_PATTERN = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


class SleepAdvisorError(ValueError):
    """Raised when an entry or date format is invalid."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SleepEntry:
    """A record of one sleep period (typically night sleep)."""

    date: str                  # ISO date: YYYY-MM-DD (morning of wake up)
    bed_time: str              # 24-hr format HH:MM (e.g. "23:00")
    wake_time: str             # 24-hr format HH:MM (e.g. "07:00")
    duration_hours: float      # Total hours slept (e.g. 8.0)
    quality: str = "good"      # "good", "fair", "poor"
    notes: str = ""            # Optional personal notes (e.g. "interrupted sleep")


@dataclass(frozen=True)
class ScorePenalty:
    """A single deducted penalty in the readiness score breakdown."""

    category: str              # Category name (e.g. "Last night deficit")
    points_lost: int           # Points deducted (positive integer)
    description: str           # Contextual explanation


@dataclass(frozen=True)
class ReadinessResult:
    """Outcome of a travel readiness evaluation."""

    score: int                           # Clamped 0 to 100
    verdict: str                         # SAFE, CAUTION, UNSAFE
    color: str                           # Hex badge color
    penalties: Tuple[ScorePenalty, ...]  # Detailed deductions
    advice: Tuple[str, ...]              # Actionable guidance
    last_night_hours: Optional[float]    # Hours slept last night
    sleep_debt_hours: float              # Cumulative debt across recent days
    hours_awake: float                   # Hours elapsed since wake up
    check_time: str                      # Time evaluated (ISO string)


# ---------------------------------------------------------------------------
# Time & Duration Helpers
# ---------------------------------------------------------------------------
def parse_time_to_minutes(time_str: str) -> int:
    """Convert an 'HH:MM' string to minutes from midnight."""
    match = _TIME_PATTERN.match(str(time_str).strip())
    if not match:
        raise SleepAdvisorError(f"Time '{time_str}' must be in 24-hour HH:MM format.")
    h, m = int(match.group(1)), int(match.group(2))
    return h * 60 + m


def calculate_sleep_duration(bed_time: str, wake_time: str) -> float:
    """
    Calculate duration in hours between bed_time and wake_time.
    Supports overnight sleep crossing midnight (e.g. 23:00 to 07:00 -> 8.0 hrs).
    """
    bed_min = parse_time_to_minutes(bed_time)
    wake_min = parse_time_to_minutes(wake_time)

    if wake_min >= bed_min:
        total_min = wake_min - bed_min
    else:
        # Crosses midnight
        total_min = (1440 - bed_min) + wake_min

    return round(total_min / 60.0, 2)


def validate_iso_date(date_str: str) -> str:
    """Validate and return YYYY-MM-DD string."""
    clean = str(date_str).strip()
    if not _DATE_PATTERN.match(clean):
        raise SleepAdvisorError(f"Date '{date_str}' must be in YYYY-MM-DD format.")
    try:
        datetime.strptime(clean, "%Y-%m-%d")
    except ValueError as exc:
        raise SleepAdvisorError(f"Invalid calendar date '{date_str}': {exc}")
    return clean


# ---------------------------------------------------------------------------
# Sleep Log Book
# ---------------------------------------------------------------------------
class SleepLogBook:
    """Loads, validates, and atomically persists the driver's sleep logs."""

    def __init__(self, path: Union[str, Path] = config.SLEEP_LOG_FILE) -> None:
        self.path = Path(path)
        self._entries: List[SleepEntry] = []
        self.load_warning: Optional[str] = None
        self.load()

    @property
    def entries(self) -> List[SleepEntry]:
        """A copy of all sleep entries, sorted by date descending (newest first)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def add_entry(
        self,
        date: str,
        bed_time: str,
        wake_time: str,
        quality: str = "good",
        notes: str = "",
        duration_hours: Optional[float] = None,
    ) -> SleepEntry:
        """
        Validate, add (or update), and save a sleep record.
        If an entry for this date already exists, it is updated.
        """
        clean_date = validate_iso_date(date)
        parse_time_to_minutes(bed_time)
        parse_time_to_minutes(wake_time)

        if duration_hours is None:
            duration = calculate_sleep_duration(bed_time, wake_time)
        else:
            duration = round(float(duration_hours), 2)
            if duration <= 0.0 or duration > 24.0:
                raise SleepAdvisorError("Duration must be between 0.1 and 24.0 hours.")

        quality_clean = str(quality).strip().lower()
        if quality_clean not in ("good", "fair", "poor"):
            quality_clean = "good"

        entry = SleepEntry(
            date=clean_date,
            bed_time=str(bed_time).strip(),
            wake_time=str(wake_time).strip(),
            duration_hours=duration,
            quality=quality_clean,
            notes=str(notes or "").strip()[:200],
        )

        # Replace existing entry for the same date if present
        self._entries = [e for e in self._entries if e.date != clean_date]
        self._entries.append(entry)
        # Keep sorted by date descending
        self._entries.sort(key=lambda e: e.date, reverse=True)
        self.save()
        return entry

    def remove_entry(self, date: str) -> bool:
        """Remove the entry for the specified date. Returns True if removed."""
        clean_date = validate_iso_date(date)
        kept = [e for e in self._entries if e.date != clean_date]
        if len(kept) == len(self._entries):
            return False
        self._entries = kept
        self.save()
        return True

    def get_entry(self, date: str) -> Optional[SleepEntry]:
        """Get the entry for a specific date, or None."""
        clean_date = validate_iso_date(date)
        for e in self._entries:
            if e.date == clean_date:
                return e
        return None

    def get_last_entry(self) -> Optional[SleepEntry]:
        """Return the most recent sleep entry recorded."""
        return self._entries[0] if self._entries else None

    def get_recent_entries(self, limit: int = config.SLEEP_DEBT_DAYS) -> List[SleepEntry]:
        """Return the most recent `limit` entries."""
        return self._entries[:limit]

    # -----------------------------------------------------------------------
    # Disk Persistence
    # -----------------------------------------------------------------------
    def load(self) -> None:
        """Read sleep log from disk. Missing file begins empty; corrupt file is set aside."""
        self._entries = []
        self.load_warning = None
        if not self.path.exists():
            return

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Sleep log file must contain a JSON list")

            loaded = []
            for item in raw:
                date = validate_iso_date(item["date"])
                bed = item["bed_time"]
                wake = item["wake_time"]
                dur = float(item.get("duration_hours") or calculate_sleep_duration(bed, wake))
                qual = str(item.get("quality", "good")).lower()
                notes = str(item.get("notes", ""))
                loaded.append(
                    SleepEntry(
                        date=date,
                        bed_time=bed,
                        wake_time=wake,
                        duration_hours=dur,
                        quality=qual,
                        notes=notes,
                    )
                )
            loaded.sort(key=lambda e: e.date, reverse=True)
            self._entries = loaded
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._entries = []
            self._set_aside_bad_file()
            self.load_warning = (
                f"Sleep log file could not be read ({exc}). "
                "Renamed to sleep_log.json.bad and started fresh."
            )

    def save(self) -> None:
        """Write the entries to disk atomically via a temporary file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "date": e.date,
                "bed_time": e.bed_time,
                "wake_time": e.wake_time,
                "duration_hours": e.duration_hours,
                "quality": e.quality,
                "notes": e.notes,
            }
            for e in self._entries
        ]

        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_name = tmp.name
                json.dump(data, tmp, indent=2, ensure_ascii=False)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, self.path)
        except OSError:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise

    def _set_aside_bad_file(self) -> None:
        bad = self.path.with_name(self.path.name + ".bad")
        try:
            os.replace(self.path, bad)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Readiness Scorer
# ---------------------------------------------------------------------------
class ReadinessScorer:
    """Evaluates whether it is safe to drive based on sleep history and circadian timing."""

    def __init__(
        self,
        target_hours: float = config.SLEEP_TARGET_HOURS,
        deficit_penalty_per_hour: float = config.SLEEP_DEFICIT_PENALTY_PER_HOUR,
        debt_days: int = config.SLEEP_DEBT_DAYS,
        debt_penalty_per_hour: float = config.SLEEP_DEBT_PENALTY_PER_HOUR,
        awake_free_hours: float = config.AWAKE_FREE_HOURS,
        awake_penalty_per_hour: float = config.AWAKE_PENALTY_PER_HOUR,
        night_start: int = config.NIGHT_START_HOUR,
        night_end: int = config.NIGHT_END_HOUR,
        night_penalty: int = config.NIGHT_PENALTY,
        dip_start: int = config.AFTERNOON_DIP_START_HOUR,
        dip_end: int = config.AFTERNOON_DIP_END_HOUR,
        dip_penalty: int = config.AFTERNOON_DIP_PENALTY,
        safe_min: int = config.SAFE_SCORE_MIN,
        caution_min: int = config.CAUTION_SCORE_MIN,
    ) -> None:
        self.target_hours = target_hours
        self.deficit_penalty_per_hour = deficit_penalty_per_hour
        self.debt_days = debt_days
        self.debt_penalty_per_hour = debt_penalty_per_hour
        self.awake_free_hours = awake_free_hours
        self.awake_penalty_per_hour = awake_penalty_per_hour
        self.night_start = night_start
        self.night_end = night_end
        self.night_penalty = night_penalty
        self.dip_start = dip_start
        self.dip_end = dip_end
        self.dip_penalty = dip_penalty
        self.safe_min = safe_min
        self.caution_min = caution_min

    def evaluate(
        self,
        entries: Sequence[SleepEntry],
        check_datetime: Optional[datetime] = None,
        wake_datetime: Optional[datetime] = None,
    ) -> ReadinessResult:
        """
        Compute readiness score and recommendations.

        Args:
            entries: List of sleep entries (newest first).
            check_datetime: The proposed driving time (defaults to datetime.now()).
            wake_datetime: Optional explicit wake up time; if omitted, inferred from the
                           most recent entry and check_datetime.
        """
        now = check_datetime if check_datetime is not None else datetime.now()
        penalties: List[ScorePenalty] = []
        advice: List[str] = []

        last_night_hours: Optional[float] = None
        sleep_debt_hours = 0.0

        # 1. Last night's sleep deficit
        if entries:
            last_entry = entries[0]
            last_night_hours = last_entry.duration_hours
            if last_night_hours < self.target_hours:
                deficit = self.target_hours - last_night_hours
                points = int(round(deficit * self.deficit_penalty_per_hour))
                penalties.append(
                    ScorePenalty(
                        category="Sleep Deficit (Last Night)",
                        points_lost=points,
                        description=(
                            f"{last_night_hours:.1f}h slept vs {self.target_hours:.1f}h target "
                            f"({deficit:.1f}h deficit, -{points} pts)"
                        ),
                    )
                )
                advice.append(
                    f"You slept {last_night_hours:.1f}h last night ({deficit:.1f}h short). "
                    "A 20-minute power nap before driving will significantly boost alertness."
                )

            # Poor quality penalty modifier
            if last_entry.quality == "poor":
                penalties.append(
                    ScorePenalty(
                        category="Poor Sleep Quality",
                        points_lost=10,
                        description="Reported disrupted or poor sleep quality (-10 pts)",
                    )
                )
                advice.append(
                    "Poor sleep quality reported. Expect faster fatigue onset; take more frequent breaks."
                )

            # 2. Accumulated sleep debt (prior days)
            prior_entries = entries[1:self.debt_days]
            if prior_entries:
                for entry in prior_entries:
                    if entry.duration_hours < self.target_hours:
                        sleep_debt_hours += (self.target_hours - entry.duration_hours)

                if sleep_debt_hours > 0.0:
                    debt_points = int(round(sleep_debt_hours * self.debt_penalty_per_hour))
                    penalties.append(
                        ScorePenalty(
                            category="Cumulative Sleep Debt",
                            points_lost=debt_points,
                            description=(
                                f"{sleep_debt_hours:.1f}h total deficit over the past "
                                f"{len(prior_entries)} prior night(s) (-{debt_points} pts)"
                            ),
                        )
                    )
                    advice.append(
                        f"Cumulative sleep debt of {sleep_debt_hours:.1f}h across recent days. "
                        "Avoid highway driving alone."
                    )
        else:
            advice.append("No sleep records logged yet. Log your recent sleep for an accurate assessment.")

        # 3. Hours awake
        inferred_wake = wake_datetime
        if inferred_wake is None and entries:
            last_entry = entries[0]
            try:
                wake_h, wake_m = (int(x) for x in last_entry.wake_time.split(":"))
                # Build wake datetime assuming wake occurred on entry date
                entry_dt = datetime.strptime(last_entry.date, "%Y-%m-%d")
                inferred_wake = entry_dt.replace(hour=wake_h, minute=wake_m, second=0, microsecond=0)
            except Exception:
                inferred_wake = None

        hours_awake = 0.0
        if inferred_wake is not None:
            diff_sec = (now - inferred_wake).total_seconds()
            if diff_sec > 0:
                hours_awake = round(diff_sec / 3600.0, 1)

            if hours_awake > self.awake_free_hours:
                excess_awake = hours_awake - self.awake_free_hours
                awake_points = int(round(excess_awake * self.awake_penalty_per_hour))
                penalties.append(
                    ScorePenalty(
                        category="Extended Wakefulness",
                        points_lost=awake_points,
                        description=(
                            f"Awake for {hours_awake:.1f}h (> {self.awake_free_hours:.0f}h limit, "
                            f"-{awake_points} pts)"
                        ),
                    )
                )
                advice.append(
                    f"You have been awake for {hours_awake:.1f}h. Wakefulness over 16h produces cognitive "
                    "impairment similar to legal intoxication. Driving is not recommended."
                )

        # 4. Circadian Window
        hour = now.hour
        if self.night_start <= hour < self.night_end:
            penalties.append(
                ScorePenalty(
                    category="High-Risk Night Window",
                    points_lost=self.night_penalty,
                    description=f"Driving between midnight and 6 AM (-{self.night_penalty} pts)",
                )
            )
            advice.append(
                "Midnight to 6 AM is the peak biological window for severe fatigue crashes. "
                "Delay trip until daylight if possible."
            )
        elif self.dip_start <= hour < self.dip_end:
            penalties.append(
                ScorePenalty(
                    category="Afternoon Circadian Dip",
                    points_lost=self.dip_penalty,
                    description=f"Driving during 1 PM - 3 PM biological dip (-{self.dip_penalty} pts)",
                )
            )
            advice.append(
                "Early afternoon circadian dip (1 PM - 3 PM). Keep vehicle well ventilated and stay hydrated."
            )

        # Total points deduction
        total_deduction = sum(p.points_lost for p in penalties)
        final_score = max(0, min(100, 100 - total_deduction))

        # Determine verdict
        if final_score >= self.safe_min:
            verdict = VERDICT_SAFE
            if not advice:
                advice.append("Sleep profile indicates optimal alertness. Take a 15-minute break every 2 hours.")
        elif final_score >= self.caution_min:
            verdict = VERDICT_CAUTION
        else:
            verdict = VERDICT_UNSAFE

        return ReadinessResult(
            score=final_score,
            verdict=verdict,
            color=VERDICT_COLORS[verdict],
            penalties=tuple(penalties),
            advice=tuple(advice),
            last_night_hours=last_night_hours,
            sleep_debt_hours=round(sleep_debt_hours, 2),
            hours_awake=hours_awake,
            check_time=now.strftime("%Y-%m-%d %H:%M"),
        )


# ---------------------------------------------------------------------------
# Optional Anthropic Chat Client
# ---------------------------------------------------------------------------
class AnthropicChatClient:
    """
    Standard-library HTTP client for the Anthropic Messages API.
    Used for conversational fatigue and sleep health queries.
    Sanitizes API keys so credentials are never leaked.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = config.CHAT_MODEL,
        max_tokens: int = config.CHAT_MAX_TOKENS,
        timeout: float = 12.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        raw_key = api_key if api_key is not None else os.environ.get(config.ANTHROPIC_API_KEY_ENV, "")
        self._api_key = str(raw_key).strip()
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._opener = opener

    @property
    def configured(self) -> bool:
        """Returns True if a valid-looking API key is present."""
        return bool(self._api_key and len(self._api_key) > 10)

    def ask(
        self,
        user_message: str,
        context_prompt: str = "",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[bool, str]:
        """
        Send a query to the Claude Messages API.

        Returns:
            (ok: bool, reply_or_error: str)
        """
        if not self.configured:
            return False, "Anthropic API key is not configured. Set the ANTHROPIC_API_KEY environment variable."

        system_instruction = (
            "You are an expert sleep health and driver fatigue advisor assisting a motorist. "
            "Give clear, direct, and actionable safety guidance. "
            "Prioritize driver safety above convenience."
        )
        if context_prompt:
            system_instruction += f"\n\nCurrent Driver Context:\n{context_prompt}"

        messages: List[Dict[str, str]] = []
        if history:
            # Keep up to configured history turns
            for turn in history[-config.CHAT_HISTORY_TURNS:]:
                if "role" in turn and "content" in turn:
                    messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": user_message[:2000]})

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_instruction,
            "messages": messages,
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "User-Agent": "driver-drowsiness-system/1.0",
            },
            method="POST",
        )

        try:
            with self._opener(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read(65536).decode("utf-8"))
                text_content = ""
                for block in resp_data.get("content", []):
                    if block.get("type") == "text":
                        text_content += block.get("text", "")
                return True, text_content.strip()
        except urllib.error.HTTPError as exc:
            return False, self._clean_error(f"HTTP {exc.code}: {exc.reason}")
        except Exception as exc:
            return False, self._clean_error(f"API request failed: {exc}")

    def _clean_error(self, msg: str) -> str:
        """Strip API key from error text."""
        if self._api_key:
            msg = msg.replace(self._api_key, "[API_KEY]")
        return msg[:200]


# ---------------------------------------------------------------------------
# High-Level Sleep Advisor Interface
# ---------------------------------------------------------------------------
class SleepAdvisor:
    """Unified coordinator connecting sleep persistence, readiness scoring, and advice."""

    def __init__(
        self,
        log_book: Optional[SleepLogBook] = None,
        scorer: Optional[ReadinessScorer] = None,
        chat_client: Optional[AnthropicChatClient] = None,
    ) -> None:
        self.log_book = log_book if log_book is not None else SleepLogBook()
        self.scorer = scorer if scorer is not None else ReadinessScorer()
        self.chat_client = chat_client if chat_client is not None else AnthropicChatClient()

    def check_readiness(
        self,
        target_datetime: Optional[datetime] = None,
        wake_datetime: Optional[datetime] = None,
    ) -> ReadinessResult:
        """Calculate the current driving readiness result."""
        return self.scorer.evaluate(
            entries=self.log_book.entries,
            check_datetime=target_datetime,
            wake_datetime=wake_datetime,
        )

    def log_sleep(
        self,
        date: str,
        bed_time: str,
        wake_time: str,
        quality: str = "good",
        notes: str = "",
        duration_hours: Optional[float] = None,
    ) -> SleepEntry:
        """Log a new sleep entry."""
        return self.log_book.add_entry(
            date=date,
            bed_time=bed_time,
            wake_time=wake_time,
            quality=quality,
            notes=notes,
            duration_hours=duration_hours,
        )

    def ask_advice(self, user_question: str, target_datetime: Optional[datetime] = None) -> str:
        """Query the advisor. Uses Claude if configured; otherwise provides rule-based offline guidance."""
        readiness = self.check_readiness(target_datetime=target_datetime)
        context = (
            f"Readiness Score: {readiness.score}/100 ({readiness.verdict})\n"
            f"Hours slept last night: {readiness.last_night_hours or 'Unknown'}\n"
            f"Recent sleep debt: {readiness.sleep_debt_hours}h\n"
            f"Hours awake: {readiness.hours_awake}h\n"
            f"Key recommendations: {'; '.join(readiness.advice)}"
        )

        if self.chat_client.configured:
            ok, reply = self.chat_client.ask(user_question, context_prompt=context)
            if ok:
                return reply

        # Offline rule-based fallback
        lines = [
            f"Travel Readiness: {readiness.verdict} ({readiness.score}/100)",
            "Safety Recommendations:",
        ]
        for a in readiness.advice:
            lines.append(f" - {a}")
        return "\n".join(lines)
