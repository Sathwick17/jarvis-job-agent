"""
Executes matched field values on the real page — no LLM involved.

Takes the output of jarvis.field_matcher and actually fills the form:
types text, uploads the resume file, selects dropdown options. Every
action here is deterministic — the LLM's only role in the pipeline is
answering the fields field_matcher couldn't resolve (see agent.py).
"""

from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import Page

from jarvis.field_matcher import MatchedField


@dataclass
class FillResult:
    element_id: str
    label: str
    ok: bool
    detail: str


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

            # A React-Select-style searchable combobox (common for Country/
            # Location on Greenhouse) looks like a text input but a plain
            # fill() doesn't register as a real selection — the widget needs
            # actual keystrokes plus clicking a rendered option, and even
            # then confirming the value stuck requires reading a different
            # element than the input itself. Not yet handled reliably;
            # flag it rather than report a false success.
            role = await locator.get_attribute("role")
            if role == "combobox":
                results.append(
                    FillResult(
                        field.element_id,
                        field.label,
                        False,
                        "skipped: searchable dropdown widget not yet supported, needs manual review",
                    )
                )
                continue

            # plain text input or textarea
            await locator.fill(m.value)
            results.append(FillResult(field.element_id, field.label, True, f"filled {m.value[:40]!r}"))

        except Exception as e:
            results.append(FillResult(field.element_id, field.label, False, f"error: {e}"))

    return results
