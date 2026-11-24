"""
scratch_chatbot_test.py
Unit tests for FatigueChatBot offline expert rules, history tracking, and API fallbacks.
"""

import tempfile
from pathlib import Path

from src.chatbot import SUGGESTED_PROMPTS, FatigueChatBot
from src.sleep_advisor import AnthropicChatClient, SleepAdvisor, SleepLogBook

# 1. Suggested prompts are available
assert len(SUGGESTED_PROMPTS) >= 4
assert any("power nap" in p.lower() for p in SUGGESTED_PROMPTS)

# 2. Offline expert mode (no API key configured)
bot = FatigueChatBot(api_key="")
assert bot.is_online_mode is False
assert len(bot.history) == 0

# Test 2a: Power nap query
reply_nap = bot.ask("How long should a power nap be?")
assert "15–20 Minute Rule" in reply_nap or "sweet spot" in reply_nap
assert "sleep inertia" in reply_nap.lower()
assert len(bot.history) == 2
assert bot.history[0]["role"] == "user"
assert bot.history[1]["role"] == "assistant"

# Test 2b: Caffeine nap query
reply_caff = bot.ask("What is a caffeine nap?")
assert "adenosine" in reply_caff.lower() or "caffeine" in reply_caff.lower()
assert len(bot.history) == 4

# Test 2c: Microsleep warning signs
reply_micro = bot.ask("What are the symptoms of microsleep?")
assert "microsleep" in reply_micro.lower()
assert "warning signs" in reply_micro.lower()

# Test 2d: Window roll down / myth
reply_window = bot.ask("Can rolling the window down keep me awake?")
assert "temporary" in reply_window.lower() or "myth" in reply_window.lower()

# Test 2e: Night driving preparation
reply_night = bot.ask("How to prepare for a night drive?")
assert "circadian" in reply_night.lower() or "2:00 am" in reply_night.lower()

# Test 2f: General query fallback
reply_gen = bot.ask("Hello, what do you do?")
assert "Driver Fatigue Safety Guidance" in reply_gen

# Test 2g: Empty query handling
reply_empty = bot.ask("   ")
assert "Please enter a question" in reply_empty

# 3. History clearing
bot.clear_history()
assert len(bot.history) == 0

# 4. Integration with SleepAdvisor
with tempfile.TemporaryDirectory() as tmp:
    book = SleepLogBook(Path(tmp) / "sleep.json")
    book.add_entry("2026-11-20", "23:00", "07:00", "good")
    advisor = SleepAdvisor(log_book=book)
    bot_with_advisor = FatigueChatBot(api_key="", advisor=advisor)

    reply_adv = bot_with_advisor.ask("Tell me about power naps")
    assert "Your Current Status:" in reply_adv
    assert "Readiness Score:" in reply_adv

# 5. Simulated API failure fallback
class FailingClient(AnthropicChatClient):
    @property
    def configured(self):
        return True

    def ask(self, user_message, context_prompt="", history=None):
        return False, "Rate limit exceeded (fake error)"

failing_bot = FatigueChatBot(chat_client=FailingClient(api_key="sk-test"))
assert failing_bot.is_online_mode is True
reply_fallback = failing_bot.ask("What is a power nap?")
assert "Offline Expert Mode" in reply_fallback
assert "15–20 Minute Rule" in reply_fallback or "sweet spot" in reply_fallback

print("All chatbot tests passed")
