"""
Maps a form field's label to a profile key — in code, no LLM.

Every field that matches a known pattern here gets filled deterministically.
Anything left unmatched is genuinely ambiguous (a custom free-text question)
and gets handed to the LLM in agent.py, one field at a time — not by
re-reading the whole page, just that field's own label and required-ness.
"""

import re
from dataclasses import dataclass

from jarvis.form_reader import FormField

# Ordered so more specific patterns are checked before generic ones
# (e.g. "linkedin" before a bare "url" pattern would ever be added).
#
# Patterns match against the field's LABEL only, and only short labels
# (see _MAX_LABEL_WORDS_FOR_KEYWORD_MATCH below) — a long sentence that
# happens to contain the word "country" (e.g. a visa-sponsorship question)
# must never be treated as a location field just because the word appears.
_PROFILE_KEY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("first_name", re.compile(r"\bfirst\s*name\b", re.IGNORECASE)),
    ("last_name", re.compile(r"\blast\s*name\b", re.IGNORECASE)),
    ("full_name", re.compile(r"^\s*(full\s*)?name\s*$", re.IGNORECASE)),
    ("email", re.compile(r"\bemail\b", re.IGNORECASE)),
    ("phone", re.compile(r"\bphone\b", re.IGNORECASE)),
    ("location", re.compile(r"^\s*(country|location|address)\s*$", re.IGNORECASE)),
    ("linkedin_url", re.compile(r"\blinkedin\b", re.IGNORECASE)),
    ("github_url", re.compile(r"\bgithub\b", re.IGNORECASE)),
    ("portfolio_url", re.compile(r"\b(portfolio|website|publications)\b", re.IGNORECASE)),
]

# Longer, sentence-style labels are questions, not simple field names — only
# match short labels by keyword to avoid false positives like a visa
# question containing the word "country".
_MAX_LABEL_WORDS_FOR_KEYWORD_MATCH = 4

# Fields no applicant profile should auto-answer — EEO/demographic
# self-identification questions are legally meant to be the applicant's own
# choice, and Greenhouse always marks these optional. Never guess these.
_SKIP_PATTERNS = re.compile(
    r"\b(gender|hispanic|latino|ethnicity|veteran|disability)\b", re.IGNORECASE
)

# Signals this isn't a fillable field at all.
_NON_FIELD_IDS = re.compile(r"recaptcha|captcha", re.IGNORECASE)


@dataclass
class MatchedField:
    field: FormField
    profile_key: str
    value: str


@dataclass
class UnmatchedField:
    field: FormField
    reason: str  # "ambiguous" | "skipped_demographic" | "captcha"


def match_fields(
    fields: list[FormField], profile: dict
) -> tuple[list[MatchedField], list[UnmatchedField]]:
    matched: list[MatchedField] = []
    unmatched: list[UnmatchedField] = []

    for field in fields:
        if _NON_FIELD_IDS.search(field.element_id):
            unmatched.append(UnmatchedField(field, "captcha"))
            continue

        if _SKIP_PATTERNS.search(field.label) or _SKIP_PATTERNS.search(field.element_id):
            unmatched.append(UnmatchedField(field, "skipped_demographic"))
            continue

        # File inputs: match by element id, not label — Greenhouse labels
        # both the resume and cover-letter uploads "Attach", so the label
        # alone can't tell them apart.
        if field.input_type == "file":
            if "resume" in field.element_id.lower():
                matched.append(MatchedField(field, "resume_path", str(profile.get("resume_path", ""))))
            else:
                unmatched.append(UnmatchedField(field, "ambiguous"))
            continue

        profile_key = _match_one(field)
        if profile_key and profile.get(profile_key):
            matched.append(MatchedField(field, profile_key, str(profile[profile_key])))
        else:
            unmatched.append(UnmatchedField(field, "ambiguous"))

    return matched, unmatched


def _match_one(field: FormField) -> str | None:
    label_word_count = len(field.label.split())
    candidates = [field.element_id]
    if label_word_count <= _MAX_LABEL_WORDS_FOR_KEYWORD_MATCH:
        # Long, sentence-style labels are custom questions, not simple
        # field names — only keyword-match short labels, checked
        # separately from element_id so the anchored patterns (^...$)
        # work correctly against the label on its own.
        candidates.append(field.label)

    for profile_key, pattern in _PROFILE_KEY_PATTERNS:
        if any(pattern.search(text) for text in candidates):
            return profile_key
    return None
