"""
Reads a job application form's fields directly via Playwright — no LLM.

This replaces asking an LLM to re-discover the page structure on every
agent step. Standard ATS forms (Greenhouse, and similarly-structured
platforms) label every field with a real <label for="..."> even when
aria-label is empty, so field identity can be read deterministically.

Each field is returned with its label text; jarvis.field_matcher then
decides, in code, which ones map to profile data and which are genuinely
ambiguous free-text questions that need an LLM to answer.
"""

from dataclasses import dataclass

from playwright.async_api import Page


@dataclass
class FormField:
    element_id: str
    tag: str  # "input", "textarea", "select"
    input_type: str  # "text", "email", "tel", "file", "" for textarea/select
    label: str
    required: bool


async def _label_for(page: Page, element_id: str) -> str:
    if not element_id:
        return ""
    label_locator = page.locator(f'label[for="{element_id}"]')
    if await label_locator.count() == 0:
        return ""
    text = await label_locator.first.inner_text()
    return text.strip()


async def read_form_fields(page: Page) -> list[FormField]:
    fields: list[FormField] = []

    for tag in ("input", "select", "textarea"):
        locator = page.locator(tag)
        count = await locator.count()
        for i in range(count):
            el = locator.nth(i)
            element_id = await el.get_attribute("id") or ""
            input_type = (await el.get_attribute("type") or "").lower()

            # skip non-field inputs: hidden fields, the phone-country search
            # box, and anything with no id (can't be reliably targeted)
            if not element_id or input_type == "hidden":
                continue
            if "search" in element_id.lower() or input_type == "search":
                continue

            # Invisible reCAPTCHA (v3/badge) leaves a hidden textarea in the
            # DOM on nearly every Greenhouse form even when there's no
            # actual visible challenge — it's not a field to fill or a
            # blocker, just background scoring. A genuinely visible CAPTCHA
            # challenge (v2 checkbox/image grid) would show up as a real,
            # visible field and should still be caught downstream.
            if "captcha" in element_id.lower() and not await el.is_visible():
                continue

            label = await _label_for(page, element_id)
            required = "*" in label or bool(await el.get_attribute("required"))

            fields.append(
                FormField(
                    element_id=element_id,
                    tag=tag,
                    input_type=input_type,
                    label=label.rstrip("*").strip(),
                    required=required,
                )
            )

    return fields
