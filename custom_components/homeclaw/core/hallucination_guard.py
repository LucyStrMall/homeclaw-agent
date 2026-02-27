"""Hallucination guard: detects model responses that claim tool results without calling tools."""

from __future__ import annotations

import re

ENTITY_ID_PATTERN = re.compile(
    r"\b(?:sensor|light|switch|media_player|climate|cover|binary_sensor|fan|lock|"
    r"vacuum|camera|automation|scene|script|input_boolean|input_number|input_select|"
    r"input_text|person|weather|zone)\.\w+"
)

HALLUCINATION_PHRASES = [
    "Sprawdzam",
    "Włączam",
    "Wyłączam",
    "Stan:",
    "jest włączony",
    "jest wyłączon",
    "temperatura wynosi",
    "aktualny stan",
    "Checking",
    "Turning on",
    "Turning off",
    "Current state",
    "is currently",
    "temperature is",
    "is on",
    "is off",
    "✅",
    "Done!",
    "Gotowe!",
    "Zrobione!",
]


def detect_hallucinated_actions(
    response_text: str,
    tool_calls_made: list[str],
) -> dict:
    if tool_calls_made:
        return {"hallucinated": False}

    entities_mentioned = ENTITY_ID_PATTERN.findall(response_text)

    phrase_found: str | None = None
    for phrase in HALLUCINATION_PHRASES:
        if phrase in response_text:
            phrase_found = phrase
            break

    if not entities_mentioned and phrase_found is None:
        return {"hallucinated": False}

    reasons = []
    if entities_mentioned:
        reasons.append(f"mentions entities: {', '.join(set(entities_mentioned))}")
    if phrase_found:
        reasons.append(f"contains action/state phrase: '{phrase_found}'")

    return {
        "hallucinated": True,
        "reason": "Response claims device state/action " + " and ".join(reasons) + " but no tools were called.",
        "entities_mentioned": list(set(entities_mentioned)),
    }


def build_correction_prompt(detection_result: dict) -> str:
    entities = detection_result.get("entities_mentioned", [])
    entity_str = ", ".join(entities) if entities else "some entities"
    return (
        f"You mentioned {entity_str} but did not call any tools. "
        "You MUST call get_entity_state, get_state, or other available tools "
        "before describing device states or performing actions. "
        "Do NOT describe states from memory or assumptions. Try again using tools."
    )
