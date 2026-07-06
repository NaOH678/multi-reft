from __future__ import annotations

import re


ANSWER_FORMAT_RE = re.compile(r"answer\s+format\s*:", re.IGNORECASE)
CHOICE_MARKER_RE = re.compile(r"\b(answer|option|ending|solution)\s*([0-9]+)\s*:", re.IGNORECASE)
EXPLICIT_LABEL_PATTERNS = [
    re.compile(
        r"(?:the\s+)?correct\s+(?:answer|response|option|ending|solution)\s*(?:is|:)\s*"
        r"((?:answer|option|ending|solution)[0-9]+|true|false)\b",
        re.IGNORECASE,
    ),
    re.compile(r"final\s+answer\s*(?:is|:)?\s*((?:answer|option|ending|solution)[0-9]+|true|false)\b", re.IGNORECASE),
    re.compile(r"(?:my\s+)?answer\s*(?:is|:)\s*((?:answer|option|ending|solution)[0-9]+|true|false)\b", re.IGNORECASE),
]
BARE_LABEL_RE = re.compile(r"\b((?:answer|option|ending|solution)[0-9]+|true|false)\b", re.IGNORECASE)


def parse_labeled_choices(question: str) -> dict[str, str]:
    text = str(question or "")
    format_match = ANSWER_FORMAT_RE.search(text)
    cutoff = format_match.start() if format_match else len(text)
    choice_region = text[:cutoff]

    matches = list(CHOICE_MARKER_RE.finditer(choice_region))
    choices: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = f"{match.group(1).lower()}{int(match.group(2))}"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else cutoff
        choice_text = re.sub(r"\s+", " ", choice_region[start:end]).strip()
        if choice_text:
            choices[label] = choice_text
    return choices


def extract_choice_label(response_text: str) -> str:
    text = str(response_text or "").strip()
    if not text:
        return ""

    for pattern in EXPLICIT_LABEL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return str(matches[-1]).strip().lower()

    matches = BARE_LABEL_RE.findall(text)
    if matches:
        return str(matches[-1]).strip().lower()
    return ""


def expand_truth_response_with_choice_text(question: str, response_text: str) -> str:
    response = str(response_text or "").strip()
    if not response:
        return response

    label = extract_choice_label(response)
    if not label or label in {"true", "false"}:
        return response

    choice_text = parse_labeled_choices(question).get(label)
    if not choice_text:
        return response

    if choice_text.lower() in response.lower():
        return response

    separator = "" if response.endswith((".", "!", "?")) else "."
    return f"{response}{separator} {choice_text}"
