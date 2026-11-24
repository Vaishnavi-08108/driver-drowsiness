"""
scratch_sleep_advisor_test.py
Comprehensive test suite for SleepLogBook, ReadinessScorer, and SleepAdvisor.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

from src import config
from src.sleep_advisor import (
    SleepAdvisor,
    SleepAdvisorError,
    SleepEntry,
    SleepLogBook,
    ReadinessScorer,
    AnthropicChatClient,
    calculate_sleep_duration,
    parse_time_to_minutes,
    validate_iso_date,
    VERDICT_SAFE,
    VERDICT_CAUTION,
    VERDICT_UNSAFE,
)

# 1. Time & duration calculations
assert parse_time_to_minutes("00:00") == 0
assert parse_time_to_minutes("07:30") == 450
assert parse_time_to_minutes("23:59") == 1439
try:
    parse_time_to_minutes("25:00")
    assert False, "Should reject invalid hour"
except SleepAdvisorError:
    pass

# Duration calculations: regular daytime & overnight crossing midnight
assert calculate_sleep_duration("22:00", "06:00") == 8.0
assert calculate_sleep_duration("23:30", "07:00") == 7.5
assert calculate_sleep_duration("01:00", "07:30") == 6.5
assert calculate_sleep_duration("14:00", "15:30") == 1.5

# Date validation
assert validate_iso_date("2026-11-20") == "2026-11-20"
try:
    validate_iso_date("20-11-2026")
    assert False, "Should reject non-ISO date"
except SleepAdvisorError:
    pass
try:
    validate_iso_date("2026-02-30")
    assert False, "Should reject invalid calendar date"
except SleepAdvisorError:
    pass


# 2. SleepLogBook CRUD & Persistence
with tempfile.TemporaryDirectory() as tmp:
    log_file = Path(tmp) / "sleep_log.json"
    book = SleepLogBook(log_file)
    assert len(book) == 0

    # Add entries
    e1 = book.add_entry("2026-11-18", "23:00", "07:00", quality="good", notes="Deep sleep")
    assert len(book) == 1
    assert e1.duration_hours == 8.0
    assert e1.quality == "good"

    # Add second entry (different date)
    e2 = book.add_entry("2026-11-19", "00:00", "06:00", quality="fair")
    assert len(book) == 2
    # Verify entries are sorted descending (newest first)
    assert book.entries[0].date == "2026-11-19"
    assert book.entries[1].date == "2026-11-18"

    # Update entry for existing date
    e2_updated = book.add_entry("2026-11-19", "01:00", "07:00", quality="poor")
    assert len(book) == 2
    assert book.get_entry("2026-11-19").quality == "poor"

    # Persistence verification (reload from file)
    book_reloaded = SleepLogBook(log_file)
    assert len(book_reloaded) == 2
    assert book_reloaded.get_last_entry().date == "2026-11-19"
    assert book_reloaded.get_entry("2026-11-18").duration_hours == 8.0

    # Remove entry
    assert book.remove_entry("2026-11-18") is True
    assert book.remove_entry("2026-11-18") is False
    assert len(book) == 1

    # Corrupt file handling: should set aside as .bad and start fresh
    log_file.write_text("{ corrupt json: [ }", encoding="utf-8")
    book_corrupt = SleepLogBook(log_file)
    assert len(book_corrupt) == 0
    assert book_corrupt.load_warning is not None
    assert (Path(tmp) / "sleep_log.json.bad").exists()


# 3. ReadinessScorer Rules & Scoring
scorer = ReadinessScorer(
    target_hours=8.0,
    deficit_penalty_per_hour=10.0,
    debt_days=5,
    debt_penalty_per_hour=5.0,
    awake_free_hours=16.0,
    awake_penalty_per_hour=5.0,
    night_start=0,
    night_end=6,
    night_penalty=20,
    dip_start=13,
    dip_end=15,
    dip_penalty=10,
    safe_min=70,
    caution_min=40,
)

# Test 3a: Perfect sleep (8h, good quality, midday drive at 10 AM, 3h awake)
entries_perfect = [SleepEntry("2026-11-20", "23:00", "07:00", 8.0, "good")]
res_perfect = scorer.evaluate(
    entries_perfect,
    check_datetime=datetime(2026, 11, 20, 10, 0),
    wake_datetime=datetime(2026, 11, 20, 7, 0),
)
assert res_perfect.score == 100
assert res_perfect.verdict == VERDICT_SAFE
assert len(res_perfect.penalties) == 0

# Test 3b: Last night deficit (5h slept vs 8h target -> 3h short -> -30 pts)
entries_deficit = [SleepEntry("2026-11-20", "02:00", "07:00", 5.0, "good")]
res_deficit = scorer.evaluate(
    entries_deficit,
    check_datetime=datetime(2026, 11, 20, 10, 0),
    wake_datetime=datetime(2026, 11, 20, 7, 0),
)
assert res_deficit.score == 70  # 100 - 30
assert res_deficit.verdict == VERDICT_SAFE
assert any("Sleep Deficit" in p.category for p in res_deficit.penalties)

# Test 3c: Deficit + poor sleep quality (-10 pts)
entries_poor = [SleepEntry("2026-11-20", "02:00", "07:00", 5.0, "poor")]
res_poor = scorer.evaluate(
    entries_poor,
    check_datetime=datetime(2026, 11, 20, 10, 0),
    wake_datetime=datetime(2026, 11, 20, 7, 0),
)
assert res_poor.score == 60  # 100 - 30 - 10
assert res_poor.verdict == VERDICT_CAUTION

# Test 3d: Cumulative debt from prior days
entries_multi = [
    SleepEntry("2026-11-20", "00:00", "08:00", 8.0, "good"),  # 0 deficit last night
    SleepEntry("2026-11-19", "02:00", "07:00", 5.0, "good"),  # 3h deficit
    SleepEntry("2026-11-18", "01:00", "07:00", 6.0, "good"),  # 2h deficit
]
# total prior debt = 5h * 5 pts/h = 25 pts lost
res_debt = scorer.evaluate(
    entries_multi,
    check_datetime=datetime(2026, 11, 20, 11, 0),
    wake_datetime=datetime(2026, 11, 20, 8, 0),
)
assert res_debt.score == 75  # 100 - 25
assert res_debt.sleep_debt_hours == 5.0

# Test 3e: Extended wakefulness (>16 hours awake)
# Wake at 6 AM, driving check at 23:00 (17h awake -> 1h excess -> -5 pts)
res_awake = scorer.evaluate(
    entries_perfect,
    check_datetime=datetime(2026, 11, 20, 23, 0),
    wake_datetime=datetime(2026, 11, 20, 6, 0),
)
assert res_awake.hours_awake == 17.0
assert res_awake.score == 95  # 100 - 5

# Test 3f: Circadian dip: 14:00 (2 PM) afternoon dip (-10 pts)
res_dip = scorer.evaluate(
    entries_perfect,
    check_datetime=datetime(2026, 11, 20, 14, 0),
    wake_datetime=datetime(2026, 11, 20, 7, 0),
)
assert res_dip.score == 90  # 100 - 10
assert any("Afternoon Circadian Dip" in p.category for p in res_dip.penalties)

# Test 3g: Circadian night window: 03:00 AM (-20 pts)
res_night = scorer.evaluate(
    entries_perfect,
    check_datetime=datetime(2026, 11, 20, 3, 0),
    wake_datetime=datetime(2026, 11, 19, 7, 0),
)
assert any("Night Window" in p.category for p in res_night.penalties)

# Test 3h: Severe combination leading to UNSAFE verdict (< 40)
# 4h sleep last night (-40), poor quality (-10), night window (-20), 10h prior debt (-50) -> clamped to 0
entries_severe = [
    SleepEntry("2026-11-20", "03:00", "07:00", 4.0, "poor"),
    SleepEntry("2026-11-19", "03:00", "07:00", 4.0, "poor"),
    SleepEntry("2026-11-18", "03:00", "07:00", 4.0, "poor"),
    SleepEntry("2026-11-17", "03:00", "07:00", 4.0, "poor"),
]
res_severe = scorer.evaluate(
    entries_severe,
    check_datetime=datetime(2026, 11, 20, 2, 0),
)
assert res_severe.score == 0
assert res_severe.verdict == VERDICT_UNSAFE


# 4. AnthropicChatClient behavior
client_empty = AnthropicChatClient(api_key="")
assert client_empty.configured is False
ok, err = client_empty.ask("Can I drive?")
assert ok is False
assert "not configured" in err

# Error sanitization test: secret key must never appear in error string
client_fake = AnthropicChatClient(api_key="sk-ant-testkey1234567890abcdef")
assert client_fake.configured is True
sanitized = client_fake._clean_error("Error at sk-ant-testkey1234567890abcdef happened")
assert "sk-ant" not in sanitized
assert "[API_KEY]" in sanitized


# 5. SleepAdvisor Facade
with tempfile.TemporaryDirectory() as tmp:
    book = SleepLogBook(Path(tmp) / "sleep.json")
    advisor = SleepAdvisor(log_book=book, scorer=scorer)
    advisor.log_sleep("2026-11-20", "23:00", "07:00", "good")
    res = advisor.check_readiness(target_datetime=datetime(2026, 11, 20, 10, 0))
    assert res.score == 100
    assert res.verdict == VERDICT_SAFE

    advice = advisor.ask_advice("Should I drive?", target_datetime=datetime(2026, 11, 20, 10, 0))
    assert "Travel Readiness: SAFE (100/100)" in advice

print("All sleep advisor tests passed")
