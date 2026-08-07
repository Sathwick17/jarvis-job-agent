"""
Answers the fields field_matcher couldn't resolve deterministically.

Each call is a single, focused LLM request scoped to ONE field's label —
not the whole page, not a multi-step loop. This is the only place in the
pipeline that calls the LLM at all, and it happens once per ambiguous
field rather than once per navigation step, which is what made the old
full-agent-loop approach slow and prone to losing track of the task.
"""

from browser_use.llm.litellm.chat import ChatLiteLLM
from browser_use.llm.messages import SystemMessage, UserMessage

from jarvis.field_matcher import UnmatchedField

_SYSTEM_PROMPT = """You are helping fill out a real job application form on behalf of a real \
applicant. You will be given one form field's label and the applicant's background. Respond \
with ONLY the text that should go in that field — no preamble, no quotation marks, no \
explanation. Keep it brief and directly responsive to what the field is actually asking. \
If the field expects a short factual answer (e.g. yes/no, a date, a number), give exactly \
that, nothing more."""


def build_question_prompt(field_label: str, profile: dict, job_context: str = "") -> str:
    return f"""Applicant background:
- Name: {profile.get('full_name', '')}
- Years of experience: {profile.get('years_of_experience', '')}
- Work authorization: {profile.get('work_authorization', '')}
- Sponsorship: {profile.get('sponsorship_required', '')}
{f"Job context: {job_context}" if job_context else ""}

Form field label: {field_label!r}

Answer for this field:"""


async def answer_field(
    llm: ChatLiteLLM, field_label: str, profile: dict, job_context: str = ""
) -> str:
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        UserMessage(content=build_question_prompt(field_label, profile, job_context)),
    ]
    result = await llm.ainvoke(messages)
    return result.completion.strip()


async def answer_unmatched_fields(
    llm: ChatLiteLLM,
    unmatched: list[UnmatchedField],
    profile: dict,
    job_context: str = "",
) -> dict[str, str]:
    """Returns {element_id: answer} for fields worth asking the LLM about.

    Skips demographic/EEO fields and CAPTCHA markers entirely — those were
    already filtered by field_matcher and must never reach the LLM.
    """
    answers: dict[str, str] = {}
    for u in unmatched:
        if u.reason != "ambiguous":
            continue
        if not u.field.label:
            continue
        answer = await answer_field(llm, u.field.label, profile, job_context)
        answers[u.field.element_id] = answer
    return answers
