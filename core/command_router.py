"""
Lightweight command routing for A.D.A.

This intentionally uses deterministic parsing for explicit app actions instead
of loading a second intent-classification model. Normal questions continue to
the primary K2 model.
"""

import re
from typing import Any, Dict, Optional, Tuple


def route_command(text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Recognize explicit supported actions; return (None, {}) for normal chat."""
    original = (text or "").strip()
    lower = original.lower()

    timer_patterns = (
        r"\b(?:set|start|create)\s+(?:a\s+)?timer\s+(?:for\s+)?(.+)",
        r"\b(?:set|start|create)\s+(?:a\s+)?(.+?)\s+timer\b",
    )
    for pattern in timer_patterns:
        match = re.search(pattern, lower)
        if match:
            return "set_timer", {"duration": match.group(1).strip(), "label": "Timer"}

    match = re.search(r"\bset\s+(?:an?\s+)?alarm\s+(?:for|at)\s+(.+)", lower)
    if match:
        return "set_alarm", {"time": match.group(1).strip(), "label": "Alarm"}

    match = re.search(r"\badd\s+(?:a\s+)?task\s+(?:to\s+)?(.+)", original, re.IGNORECASE)
    if match:
        return "add_task", {"text": match.group(1).strip()}

    match = re.search(r"\bsearch(?:\s+the\s+web)?\s+for\s+(.+)", original, re.IGNORECASE)
    if match:
        return "web_search", {"query": match.group(1).strip()}

    match = re.search(
        r"\bturn\s+(on|off|toggle|dim)\s+(?:the\s+)?(.+?)(?:\s+lights?)?$",
        lower,
    )
    if match:
        action, device = match.groups()
        return "control_light", {
            "action": action,
            "device_name": device.strip(),
        }

    # Basic calendar creation; require both a recognizable date and title.
    match = re.search(
        r"\b(?:create|add|schedule)\s+(?:a\s+)?(?:calendar\s+)?event\s+(.+?)\s+"
        r"(?:on\s+)?(today|tomorrow|\d{4}-\d{2}-\d{2}|next\s+\w+)"
        r"(?:\s+at\s+(.+))?$",
        original,
        re.IGNORECASE,
    )
    if match:
        title, date, time_value = match.groups()
        return "create_calendar_event", {
            "title": title.strip(),
            "date": date.strip(),
            "time": (time_value or "09:00").strip(),
        }

    if any(term in lower for term in (
        "what is on my schedule",
        "what's on my schedule",
        "what do i have today",
        "show my tasks",
        "system status",
        "system information",
        "what's the weather",
        "what is the weather",
    )):
        return "get_system_info", {}

    return None, {}
