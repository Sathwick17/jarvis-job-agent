"""
Answers the fields field_matcher couldn't resolve deterministically.

Each call is a single, focused LLM request scoped to ONE field's label —
not the whole page, not a multi-step loop. This is the only place in the
pipeline that calls the LLM at all, and it happens once per ambiguous
field rather than once per navigation step, which is what made the old
full-agent-loop approach slow and prone to losing track of the task.
"""

import re

from browser_use.llm.litellm.chat import ChatLiteLLM
from browser_use.llm.messages import SystemMessage, UserMessage

from jarvis.field_matcher import UnmatchedField

_SYSTEM_PROMPT = """You are helping fill out a real job application form on behalf of a real \
applicant. You will be given one form field's label and only the applicant background that is \
actually relevant to it. Respond with ONLY the text that should go in that field — no \
preamble, no quotation marks, no explanation. Keep it brief and directly responsive to what \
the field is actually asking. If the field expects a short factual answer (e.g. yes/no, a \
date, a number), give exactly that, nothing more. If no background was given because nothing \
relevant applies, give a brief, reasonable, generic answer rather than inventing specifics."""

# Rather than hand every question the applicant's entire profile and trust
# the model to ignore what's irrelevant (tested and failed — it answered a
# scheduling question with immigration status because that fact was simply
# present in the prompt), only include background that keyword-matches
# what the question is actually about. Same principle as field_matcher:
# decide relevance in code, don't rely on the LLM to self-filter.
_RELEVANCE_PATTERNS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"visa|sponsor|authoriz|work\s*permit|citizen", re.IGNORECASE),
     ["work_authorization", "sponsorship_required"]),
    (re.compile(r"experience|years|background|qualif", re.IGNORECASE),
     ["years_of_experience"]),
    (re.compile(r"why|interest|motivat|passion", re.IGNORECASE),
     ["years_of_experience"]),
]


def _relevant_background(field_label: str, profile: dict) -> dict:
    background = {}
    for pattern, keys in _RELEVANCE_PATTERNS:
        if pattern.search(field_label):
            for key in keys:
                if profile.get(key):
                    background[key] = profile[key]
    return background


def build_question_prompt(
    field_label: str, profile: dict, job_context: str = "", options: list[str] | None = None
) -> str:
    background = _relevant_background(field_label, profile)
    background_lines = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in background.items())

    options_instruction = ""
    if options:
        options_list = ", ".join(repr(o) for o in options)
        options_instruction = (
            f"\nThis field is a dropdown with EXACTLY these choices: {options_list}. "
            f"Your entire response must be one of these choices, verbatim, with nothing else added."
        )

    return f"""{f"Relevant applicant background:\n{background_lines}" if background_lines else "No specific applicant background applies to this field."}
{f"Job context: {job_context}" if job_context else ""}

Form field label: {field_label!r}
{options_instruction}
Answer for this field:"""


async def answer_field(
    llm: ChatLiteLLM,
    field_label: str,
    profile: dict,
    job_context: str = "",
    options: list[str] | None = None,
) -> str:
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        UserMessage(content=build_question_prompt(field_label, profile, job_context, options)),
    ]
    result = await llm.ainvoke(messages)
    answer = result.completion.strip()

    if options:
        # Guard against the model still adding extra words despite the
        # instruction — snap to the closest matching option rather than
        # pass through free text that a combobox can't be matched against.
        exact = next((o for o in options if o.strip().lower() == answer.lower()), None)
        if exact:
            return exact
        starts_with = next((o for o in options if answer.lower().startswith(o.strip().lower())), None)
        if starts_with:
            return starts_with

    return answer


async def answer_unmatched_fields(
    llm: ChatLiteLLM,
    unmatched: list[UnmatchedField],
    profile: dict,
    job_context: str = "",
    options_by_id: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Returns {element_id: answer} for fields worth asking the LLM about.

    Skips demographic/EEO fields and CAPTCHA markers entirely — those were
    already filtered by field_matcher and must never reach the LLM.

    options_by_id, when provided, tells the LLM the exact valid choices
    for combobox-style fields (e.g. ["Yes", "No"]) so it picks one of
    them directly instead of writing free text that then has to be
    pattern-matched against the widget's rendered options afterward.
    """
    options_by_id = options_by_id or {}
    answers: dict[str, str] = {}
    for u in unmatched:
        if u.reason != "ambiguous":
            continue
        if not u.field.label:
            continue
        options = options_by_id.get(u.field.element_id)
        answer = await answer_field(llm, u.field.label, profile, job_context, options)
        answers[u.field.element_id] = answer
    return answers
