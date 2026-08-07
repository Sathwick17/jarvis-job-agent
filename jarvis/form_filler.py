"""
Executes matched field values on the real page — no LLM involved.

Takes the output of jarvis.field_matcher and actually fills the form:
types text, uploads the resume file, selects dropdown options. Every
action here is deterministic — the LLM's only role in the pipeline is
answering the fields field_matcher couldn't resolve (see agent.py).
"""

import re
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import Page

from jarvis.field_matcher import MatchedField

_COMBOBOX_OPTION_TIMEOUT_MS = 2000


@dataclass
class FillResult:
    element_id: str
    label: str
    ok: bool
    detail: str


async def get_combobox_options(page: Page, element_id: str) -> list[str]:
    """Opens a combobox just to read its available options, without
    selecting anything — used to tell the LLM what the valid choices
    are before it answers, instead of letting it write free text that
    then has to be pattern-matched against a fixed list after the fact.
    """
    locator = page.locator(f"#{element_id}")
    options_locator = page.locator(f'[id^="react-select-{element_id}-option"]')
    try:
        await locator.click()
        await options_locator.first.wait_for(timeout=_COMBOBOX_OPTION_TIMEOUT_MS)
        options = await options_locator.all_text_contents()
    except Exception:
        options = []
    finally:
        await page.keyboard.press("Escape")
    return [o.strip() for o in options]


def _find_matching_option(answer: str, option_texts: list[str]) -> int | None:
    answer_lower = answer.strip().lower()

    # exact match first
    for i, t in enumerate(option_texts):
        if t.strip().lower() == answer_lower:
            return i

    # the answer often leads with the option word ("Yes, because...") —
    # check if the answer STARTS WITH an option, not the reverse (a short
    # option like "Yes" is very likely to be a substring of unrelated text,
    # but an answer starting with the exact option word is a strong signal)
    for i, t in enumerate(option_texts):
        option_lower = t.strip().lower()
        if answer_lower.startswith(option_lower):
            return i

    # fall back to the option appearing as a whole word anywhere in the answer
    for i, t in enumerate(option_texts):
        option_lower = t.strip().lower()
        if re.search(rf"\b{re.escape(option_lower)}\b", answer_lower):
            return i

    # Last resort: word-overlap. Handles cases like a stored profile value
    # of "Not a protected veteran" against a real option worded "I am not
    # a protected veteran" — different phrasing, same meaning, most content
    # words shared. Require most of the answer's words to appear in the
    # option (not the other way around) so a short generic option like
    # "No" doesn't win by accident against an unrelated longer answer.
    _STOPWORDS = {"a", "an", "the", "i", "am", "is", "are", "to", "of", "or", "and"}
    answer_words = {w for w in re.findall(r"[a-z0-9']+", answer_lower) if w not in _STOPWORDS}
    if answer_words:
        best_index, best_overlap = None, 0.0
        for i, t in enumerate(option_texts):
            option_words = {w for w in re.findall(r"[a-z0-9']+", t.strip().lower()) if w not in _STOPWORDS}
            if not option_words:
                continue
            overlap = len(answer_words & option_words) / len(answer_words)
            if overlap > best_overlap:
                best_index, best_overlap = i, overlap
        if best_index is not None and best_overlap >= 0.6:
            return best_index

    return None


async def fill_combobox(page: Page, element_id: str, value: str) -> FillResult:
    """Fills a React-Select-style searchable combobox (used by Greenhouse
    for Country and most Yes/No custom questions) — these render as
    <input role="combobox">, so a plain .fill() doesn't register as a
    real selection; the widget needs a click to open, then a click on
    the matching rendered option.
    """
    locator = page.locator(f"#{element_id}")
    label = ""  # caller fills this in on the returned result if needed

    try:
        await locator.click()
        options_locator = page.locator(f'[id^="react-select-{element_id}-option"]')
        try:
            await options_locator.first.wait_for(timeout=_COMBOBOX_OPTION_TIMEOUT_MS)
        except Exception:
            # No options rendered on click alone — try typing to filter
            # (needed for long lists like Country).
            await locator.press_sequentially(value, delay=40)
            await options_locator.first.wait_for(timeout=_COMBOBOX_OPTION_TIMEOUT_MS)

        option_texts = await options_locator.all_text_contents()
        match_index = _find_matching_option(value, option_texts)
        if match_index is None:
            return FillResult(element_id, label, False, f"no option matched {value!r}; saw {option_texts}")

        await options_locator.nth(match_index).click()

        # verify by reading the rendered selected text back
        selected_text = await locator.evaluate(
            'el => el.closest(".select__control")?.innerText || ""'
        )
        if selected_text.strip().lower() == option_texts[match_index].strip().lower():
            return FillResult(element_id, label, True, f"selected {option_texts[match_index]!r}")
        return FillResult(
            element_id, label, False, f"clicked {option_texts[match_index]!r} but selection didn't stick"
        )

    except Exception as e:
        return FillResult(element_id, label, False, f"error: {e}")


async def fill_matched_fields(page: Page, matched: list[MatchedField]) -> list[FillResult]:
    results: list[FillResult] = []

    for m in matched:
        field = m.field
        locator = page.locator(f"#{field.element_id}")

        try:
            if field.input_type == "file":
                path = Path(m.value)
                if not path.exists():
                    results.append(
                        FillResult(field.element_id, field.label, False, f"file not found: {m.value}")
                    )
                    continue
                await locator.set_input_files(str(path.resolve()))
                results.append(FillResult(field.element_id, field.label, True, f"uploaded {path.name}"))
                continue

            if field.tag == "select":
                await locator.select_option(label=m.value)
                results.append(FillResult(field.element_id, field.label, True, f"selected {m.value!r}"))
                continue

            # A React-Select-style searchable combobox (Country, and most
            # Yes/No custom questions on Greenhouse) looks like a text
            # input but a plain fill() doesn't register as a real
            # selection — see fill_combobox() for the click+select logic.
            role = await locator.get_attribute("role")
            if role == "combobox":
                result = await fill_combobox(page, field.element_id, m.value)
                results.append(FillResult(field.element_id, field.label, result.ok, result.detail))
                continue

            # plain text input or textarea
            await locator.fill(m.value)
            results.append(FillResult(field.element_id, field.label, True, f"filled {m.value[:40]!r}"))

        except Exception as e:
            results.append(FillResult(field.element_id, field.label, False, f"error: {e}"))

    return results
