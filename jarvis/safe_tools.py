"""
Code-level enforcement that Jarvis can never click a submit-like control.

The task prompt already tells the model not to submit, but a prompt is a
request, not a constraint — during testing the model clicked a real
"Submit application" button despite explicit instructions not to. This
module makes that boundary structural: it intercepts every click before
it reaches the browser and refuses to execute it if the target element
looks like a form-submission control, regardless of what the model decided.
"""

import re

from browser_use.browser.session import BrowserSession
from browser_use.tools.service import Tools
from browser_use.tools.utils import get_click_description
from browser_use.tools.views import ClickElementActionIndexOnly
from browser_use.agent.views import ActionResult

# Matches on the element's own text/attributes, not surrounding page copy —
# e.g. a button literally labeled "Submit", "Submit Application", "Apply Now".
_SUBMIT_PATTERN = re.compile(
    r"\b(submit|apply\s*now|send\s*application|finish\s*application)\b",
    re.IGNORECASE,
)


def _looks_like_submit(node) -> bool:
    attrs = node.attributes or {}
    if attrs.get("type", "").lower() == "submit":
        return True

    candidates = [
        attrs.get("aria-label", ""),
        attrs.get("value", ""),
        attrs.get("name", ""),
        attrs.get("id", ""),
    ]
    if hasattr(node, "get_meaningful_text_for_llm"):
        candidates.append(node.get_meaningful_text_for_llm() or "")

    blob = " ".join(candidates)
    return bool(_SUBMIT_PATTERN.search(blob))


def _describe(node) -> str:
    try:
        return get_click_description(node)
    except AttributeError:
        attrs = node.attributes or {}
        return attrs.get("aria-label") or attrs.get("value") or getattr(node, "tag_name", "element")


def build_safe_tools() -> Tools:
    """Return a Tools registry whose click action refuses submit-like elements."""
    tools = Tools()

    # Replace the registered click action with a guarded version. Re-uses
    # the same underlying _click_by_index the original registration would
    # have used, so normal clicks behave identically — only submit-like
    # targets are blocked.
    if "click" in tools.registry.registry.actions:
        del tools.registry.registry.actions["click"]

    @tools.action(
        "Click element by index.",
        param_model=ClickElementActionIndexOnly,
    )
    async def click(params: ClickElementActionIndexOnly, browser_session: BrowserSession):
        node = await browser_session.get_element_by_index(params.index)
        if node is None:
            return ActionResult(
                extracted_content=(
                    f"Element index {params.index} not available - page may have changed. "
                    "Try refreshing browser state."
                )
            )

        if _looks_like_submit(node):
            description = _describe(node)
            return ActionResult(
                error=(
                    f"BLOCKED: refused to click {description!r} — it looks like a "
                    "form-submission control (submit/apply/send application). Jarvis is "
                    "never allowed to submit an application. If the form is fully filled "
                    "in, call done with success=True and report that it is ready for "
                    "human review — do not attempt to click this or any similar button."
                )
            )

        return await tools._click_by_index(params, browser_session)

    return tools
