"""
chatbot.py
Driver Fatigue Assistant & Sleep Health Chatbot.

Provides conversational fatigue advice using Anthropic Claude Messages API
when configured with an API key, backed by an intelligent offline expert rule engine
that operates with zero external network dependencies.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src import config
from src.sleep_advisor import AnthropicChatClient, SleepAdvisor

# Common quick prompt suggestions for the UI
SUGGESTED_PROMPTS = [
    "How long should a power nap be?",
    "What is a 'caffeine nap' and how does it work?",
    "What are the warning signs of microsleep?",
    "Can rolling the window down or cold air keep me awake?",
    "What is the best way to prepare for a long night drive?",
]

# Offline expert knowledge base
_EXPERT_TOPICS = [
    {
        "keywords": ["power nap", "nap length", "how long nap", "nap duration", "napping", "sleep inertia"],
        "title": "Optimal Power Nap Duration & Sleep Inertia",
        "content": (
            "**The 15–20 Minute Rule:**\n"
            "• **15–20 minutes** is the scientifically proven sweet spot for a driver power nap. "
            "It refreshes alertness and reaction time without entering deep (slow-wave) sleep.\n"
            "• **Avoid 30–60 minute naps:** Waking from deep sleep causes **sleep inertia** "
            "(severe grogginess, disorientation, and slower reflexes) that can persist for 15–30 minutes.\n"
            "• **90-minute full cycle:** If you have adequate time, 90 minutes completes a full sleep cycle "
            "(light, deep, REM), allowing you to wake up naturally refreshed."
        ),
    },
    {
        "keywords": ["caffeine nap", "coffee nap", "caffeine", "coffee", "energy drink"],
        "title": "The 'Caffeine Nap' Technique",
        "content": (
            "**How a Caffeine Nap Works:**\n"
            "1. Rapidly consume 150–200 mg of caffeine (e.g., a cup of black coffee or espresso).\n"
            "2. Immediately close your eyes and rest for exactly **15–20 minutes**.\n"
            "3. **The Science:** Adenosine builds up in the brain and causes sleepiness. Napping naturally clears "
            "adenosine, while caffeine takes ~20–30 minutes to absorb and block adenosine receptors. "
            "Waking up just as the caffeine takes effect creates a powerful compounding boost in alertness.\n"
            "• *Caution:* Do not rely on caffeine as a substitute for real sleep; tolerance builds quickly."
        ),
    },
    {
        "keywords": ["microsleep", "micro-sleep", "warning signs", "symptoms", "falling asleep", "nodding off"],
        "title": "Microsleep Detection & Critical Warning Signs",
        "content": (
            "**What is a Microsleep?**\n"
            "A microsleep is an involuntary loss of consciousness lasting 1 to 15 seconds. "
            "At 100 km/h (62 mph), a 4-second microsleep covers over 110 meters completely uncontrolled.\n\n"
            "**Key Warning Signs:**\n"
            "• Frequent blinking, heavy eyelids, or difficulty focusing.\n"
            "• Head nodding or involuntary jerking.\n"
            "• Inability to remember the last 2–3 kilometers driven.\n"
            "• Drifting out of your lane or wandering over rumble strips.\n"
            "• Missing turnoffs, exits, or traffic signals.\n"
            "• Yawning repeatedly.\n"
            "**If you experience any of these, PULL OVER IMMEDIATELY. You cannot overpower a microsleep.**"
        ),
    },
    {
        "keywords": ["window", "cold air", "music", "radio", "slap", "gum", "temporary fix", "stay awake"],
        "title": "The Myth of Temporary Countermeasures",
        "content": (
            "**Common Myths vs. Scientific Reality:**\n"
            "• **Rolling down windows / Cold AC:** Provides a sensory burst for 3–5 minutes at best, "
            "after which the brain adapts and sleepiness returns rapidly.\n"
            "• **Loud music / Radio:** May momentarily distract, but does not improve cognitive reaction time "
            "or prevent eyelid closures.\n"
            "• **Slapping your face / Chewing gum:** False sense of security that delays safe stopping.\n\n"
            "**The Only True Cure:** Sleep. The only scientifically validated countermeasures for driver drowsiness "
            "are pulling over for a 20-minute nap, caffeine, or trading drivers."
        ),
    },
    {
        "keywords": ["night drive", "prepare", "long trip", "road trip", "highway", "schedule", "night driving"],
        "title": "Preparation for Long & Night Driving",
        "content": (
            "**Pre-Trip Checklist for Safe Long-Distance Driving:**\n"
            "• **Bank Sleep:** Ensure 7–9 hours of quality sleep for at least 2 consecutive nights before departure.\n"
            "• **Beware the Circadian Danger Zones:** Peak drowsiness occurs between **2:00 AM – 6:00 AM** and "
            "**1:00 PM – 3:00 PM**. Avoid driving through these hours whenever possible.\n"
            "• **Rule of 2 Hours:** Schedule a mandatory 15-minute rest stop every 2 hours or 160 km (100 miles).\n"
            "• **Hydration & Nutrition:** Avoid heavy, carbohydrate-heavy meals before driving (they induce postprandial "
            "somnolence / food coma). Drink water regularly.\n"
            "• **Co-Driver:** Rotate driving duties every 2 hours if traveling with passengers."
        ),
    },
    {
        "keywords": ["pull over", "emergency", "stop", "safe place", "where to stop"],
        "title": "Emergency Drowsiness Pull-Over Protocol",
        "content": (
            "**When Drowsiness Becomes Critical:**\n"
            "1. **Never stop on the highway shoulder** unless your vehicle is completely disabled. Shoulders carry "
            "severe collision risk from errant drivers.\n"
            "2. Exit the highway at the next available service area, rest stop, well-lit petrol station, or parking lot.\n"
            "3. Park in a well-lit, secure area. Lock all doors and turn off the engine.\n"
            "4. Crack the window slightly for ventilation if climate permits.\n"
            "5. Set your phone alarm for **20 minutes**.\n"
            "6. Walk around and stretch for 5 minutes after waking before resuming driving."
        ),
    },
    {
        "keywords": ["sleep debt", "debt", "catch up", "sleep deficit", "accumulated"],
        "title": "Understanding and Paying Off Sleep Debt",
        "content": (
            "**Cumulative Sleep Debt:**\n"
            "• If you need 8 hours of sleep but get only 6 hours for 4 nights, you carry an **8-hour sleep debt**.\n"
            "• Chronic sleep restriction degrades reaction time and spatial awareness to levels equivalent "
            "to 0.05% - 0.08% Blood Alcohol Concentration.\n"
            "• **Recovery:** Sleep debt cannot be fully repaid in a single long sleep. It requires 2–3 nights of "
            "consistent, unhurried rest to restore baseline psychomotor vigilance."
        ),
    },
]


class FatigueChatBot:
    """
    Driver fatigue advisory chatbot.
    Wraps Claude Messages API when configured with Anthropic API key,
    and falls back seamlessly to an offline expert knowledge engine.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        advisor: Optional[SleepAdvisor] = None,
        chat_client: Optional[AnthropicChatClient] = None,
    ) -> None:
        self.advisor = advisor if advisor is not None else SleepAdvisor()
        if chat_client is not None:
            self.client = chat_client
        else:
            self.client = AnthropicChatClient(api_key=api_key)
        self.history: List[Dict[str, str]] = []

    @property
    def is_online_mode(self) -> bool:
        """True if the Anthropic API key is configured."""
        return self.client.configured

    @property
    def is_online(self) -> bool:
        """Alias for is_online_mode."""
        return self.is_online_mode

    def chat(self, user_message: str) -> str:
        """Alias for ask."""
        return self.ask(user_message)

    def clear_history(self) -> None:
        """Clear conversation turns."""
        self.history.clear()

    def get_offline_reply(self, message: str) -> str:
        """
        Intelligent offline expert rule matcher.
        Extracts driver context and matches fatigue/sleep questions.
        """
        clean_msg = message.lower().strip()

        # Score topics by keyword matches
        best_topic = None
        best_score = 0
        for topic in _EXPERT_TOPICS:
            score = sum(1 for kw in topic["keywords"] if kw in clean_msg)
            if score > best_score:
                best_score = score
                best_topic = topic

        # Include live readiness context if available
        readiness_snippet = ""
        try:
            r = self.advisor.check_readiness()
            readiness_snippet = (
                f"\n\n---\n*Your Current Status: **{r.verdict}** (Readiness Score: {r.score}/100)*"
            )
        except Exception:
            pass

        if best_topic and best_score > 0:
            return f"### {best_topic['title']}\n\n{best_topic['content']}{readiness_snippet}"

        # General comprehensive advisory if no exact topic matched
        return (
            "### Driver Fatigue Safety Guidance\n\n"
            "I am your driver fatigue and sleep health assistant. Here are essential safety rules:\n\n"
            "1. **Never fight extreme drowsiness behind the wheel.** Reaction time is compromised before you realize it.\n"
            "2. **The 20-minute power nap** is the most effective immediate countermeasure.\n"
            "3. **Circadian risk peaks** between 2:00 AM – 6:00 AM and 1:00 PM – 3:00 PM.\n"
            "4. **Take breaks every 2 hours or 160 km** on highway trips.\n\n"
            "Feel free to ask about power naps, caffeine naps, microsleep signs, sleep debt, or road trip preparation!"
            f"{readiness_snippet}"
        )

    def ask(self, user_message: str) -> str:
        """
        Process a user question.
        Uses Anthropic Claude API if configured; otherwise uses offline expert mode.
        """
        clean_user = user_message.strip()
        if not clean_user:
            return "Please enter a question regarding fatigue or sleep health."

        # Record user turn
        self.history.append({"role": "user", "content": clean_user})

        reply: str
        if self.is_online_mode:
            # Build current driver status context
            context = ""
            try:
                r = self.advisor.check_readiness()
                context = (
                    f"Readiness Score: {r.score}/100 ({r.verdict})\n"
                    f"Last night sleep: {r.last_night_hours}h\n"
                    f"Sleep debt: {r.sleep_debt_hours}h\n"
                    f"Hours awake: {r.hours_awake}h\n"
                    f"Recommendations: {'; '.join(r.advice)}"
                )
            except Exception:
                pass

            ok, response = self.client.ask(
                clean_user,
                context_prompt=context,
                history=self.history[:-1],
            )
            if ok:
                reply = response
            else:
                # Fallback to offline on API failure, noting the fallback
                offline_ans = self.get_offline_reply(clean_user)
                reply = f"*(Offline Expert Mode — API unavailable: {response})*\n\n{offline_ans}"
        else:
            reply = self.get_offline_reply(clean_user)

        # Record assistant turn
        self.history.append({"role": "assistant", "content": reply})

        # Trim history to configured turns
        max_turns = config.CHAT_HISTORY_TURNS * 2
        if len(self.history) > max_turns:
            self.history = self.history[-max_turns:]

        return reply
