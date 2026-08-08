"""
The structured result Jarvis returns after attempting to fill a form.

This is the actual approval-gate contract: Jarvis never decides an
application is "done" in a way that leads anywhere but here. Every run
ends in exactly one ApplyResult, and the caller (a CLI today; n8n or a
dashboard later) decides what happens next — there is no code path,
here or in agent.py, that submits on Jarvis's own authority.
"""

from dataclasses import dataclass, field
from enum import Enum


class ApplyStatus(str, Enum):
    READY_FOR_REVIEW = "ready_for_review"  # form filled (or as filled as it gets), human should look it over
    FLAGGED = "flagged"                     # a real blocker (CAPTCHA, not an application form, etc.) — stopped early
    ERROR = "error"                         # something broke unexpectedly


@dataclass
class FieldOutcome:
    element_id: str
    label: str
    ok: bool
    detail: str
    source: str  # "matched" (filled by code) | "llm" (LLM answered) | "skipped"


@dataclass
class ApplyResult:
    status: ApplyStatus
    url: str
    reason: str = ""  # human-readable summary, esp. for FLAGGED/ERROR
    fields_filled: list[FieldOutcome] = field(default_factory=list)
    fields_skipped: list[FieldOutcome] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for f in self.fields_filled if f.ok)

    @property
    def needs_attention_count(self) -> int:
        return sum(1 for f in self.fields_filled if not f.ok) + len(self.fields_skipped)
