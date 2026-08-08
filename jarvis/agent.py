"""
Part 2/3: Jarvis's form-filling pipeline and approval-gate contract.

Deterministic-first: reads the real form fields via Playwright, fills
whatever matches the applicant's profile in code (no LLM), and only
calls the LLM for the handful of fields that are genuinely ambiguous
(custom free-text questions). This replaced an earlier full-agent-loop
design where an LLM re-interpreted the entire page on every step — that
approach was slow (3+ min/step locally), and in testing hallucinated
fake URLs, got stuck in scroll loops, and once clicked a real "Submit
application" button despite explicit instructions not to.

Jarvis never submits: fill_application() has no code path that clicks
anything after fields are filled — it always ends by returning an
ApplyResult (see jarvis/apply_result.py) and leaving the browser open
for a human (or, later, a dashboard) to review and submit manually.
That's a stronger guarantee than a runtime check (see jarvis/safe_tools.py,
still used by the older LLM-loop approach) because there's simply no
click-executing code left to guard. The only click this pipeline ever
performs is "Apply" on a listing page, before any filling happens.

fill_application() does the work and returns a result; it does not
manage the browser's lifecycle or print anything — that's the caller's
job (see main() below for the CLI version, or a future n8n/dashboard
caller). This split is what makes the approval gate a real contract
instead of something baked into a terminal script.

Uses Qwen, served locally by Ollama — no API keys, no cloud calls.

Usage:
    python -m jarvis.agent <job_posting_url>
"""

import asyncio
import sys
from pathlib import Path

from playwright.async_api import Browser, Page, async_playwright

from browser_use.llm.litellm.chat import ChatLiteLLM

from jarvis.apply_result import ApplyResult, ApplyStatus, FieldOutcome
from jarvis.field_matcher import match_fields
from jarvis.form_filler import fill_combobox, fill_matched_fields, get_combobox_options
from jarvis.form_reader import read_form_fields
from jarvis.profile import load_profile
from jarvis.question_answerer import answer_unmatched_fields

OLLAMA_BASE_URL = "http://localhost:11434"
# qwen2.5:3b is faster but unreliable — it hallucinated an unrelated task
# (tried to print the page as a PDF) instead of following instructions.
# qwen3:8b's default 4096-token context was too small (truncation likely
# contributed to hallucinated URLs/goals). qwen3:8b-32k fixed that but its
# KV cache consumed ~7.4GB RAM, leaving the 18GB machine starved enough
# that the model started emitting degenerate repeated-token output.
# qwen3:8b-8k is sized to what this task actually needs — and now that
# the LLM only answers a handful of individual fields (not the whole
# page every step), it's called briefly and occasionally, not for
# minutes at a stretch.
MODEL_NAME = "ollama/qwen3:8b-8k"

_PAGE_LOAD_WAIT_MS = 2500
# A combobox with more options than this (e.g. a full country list, ~250
# items) isn't worth asking a small local model to pick from — too many
# tokens, too much room for a wrong pick. Left for manual review instead.
_MAX_COMBOBOX_OPTIONS_FOR_LLM = 15


async def _click_apply_if_present(page: Page) -> bool:
    """If we're on a job listing page (not yet the application form),
    click the "Apply" link/button. Returns True if a click happened."""
    apply_locator = page.get_by_text("Apply", exact=False).first
    if await apply_locator.count() == 0:
        return False

    box = await apply_locator.bounding_box()
    if box is None:  # not visible/clickable
        return False

    await apply_locator.click(timeout=5000)
    await page.wait_for_timeout(_PAGE_LOAD_WAIT_MS)
    return True


async def fill_application(page: Page, url: str) -> ApplyResult:
    """Fills out a job application form on an already-open page.

    Does the work and returns a structured result — never prints,
    never blocks on input, never closes the page. The browser's
    lifecycle (open/close) and any human-facing presentation are the
    caller's responsibility; see main() for the CLI version.
    """
    profile = load_profile()
    fields_filled: list[FieldOutcome] = []
    fields_skipped: list[FieldOutcome] = []

    resume_path = Path(profile.get("resume_path", ""))
    resume_missing = bool(profile.get("resume_path")) and not resume_path.exists()

    await page.goto(url, timeout=30000)
    await page.wait_for_timeout(_PAGE_LOAD_WAIT_MS)

    fields = await read_form_fields(page)
    if len(fields) < 3:
        clicked = await _click_apply_if_present(page)
        if clicked:
            fields = await read_form_fields(page)
        if len(fields) < 3:
            return ApplyResult(
                status=ApplyStatus.FLAGGED,
                url=url,
                reason=f"Only found {len(fields)} field(s) — this may not be a direct "
                       "application form URL.",
            )

    matched, unmatched = match_fields(fields, profile)
    if any(u.reason == "captcha" for u in unmatched):
        return ApplyResult(
            status=ApplyStatus.FLAGGED,
            url=url,
            reason="CAPTCHA detected on this form. Jarvis does not attempt to "
                   "solve or bypass CAPTCHAs by design.",
        )

    fill_results = await fill_matched_fields(page, matched)
    for r in fill_results:
        fields_filled.append(FieldOutcome(r.element_id, r.label, r.ok, r.detail, source="matched"))

    # Greenhouse (and likely other ATSs) conditionally reveal new fields
    # after a combobox is answered — e.g. "Please identify your race"
    # only appears once "Are you Hispanic/Latino?" has been answered.
    # Re-scan once for anything that appeared as a side effect of
    # filling and fold it into the same matched/unmatched flow.
    known_ids = {f.element_id for f in fields}
    fields_after_fill = await read_form_fields(page)
    newly_revealed = [f for f in fields_after_fill if f.element_id not in known_ids]
    if newly_revealed:
        new_matched, new_unmatched = match_fields(newly_revealed, profile)
        if new_matched:
            new_fill_results = await fill_matched_fields(page, new_matched)
            for r in new_fill_results:
                fields_filled.append(FieldOutcome(r.element_id, r.label, r.ok, r.detail, source="matched"))
        unmatched += new_unmatched

    # File inputs (e.g. an optional cover-letter upload we have no file
    # for) are never something the LLM can "answer" with text.
    file_inputs = [u for u in unmatched if u.reason == "ambiguous" and u.field.input_type == "file"]
    for u in file_inputs:
        fields_skipped.append(
            FieldOutcome(u.field.element_id, u.field.label, False, "no matching file to attach", source="skipped")
        )

    for u in unmatched:
        if u.reason == "skipped_demographic":
            fields_skipped.append(
                FieldOutcome(
                    u.field.element_id, u.field.label, False,
                    "never auto-answered without full EEO opt-in", source="skipped",
                )
            )

    ambiguous = [
        u for u in unmatched
        if u.reason == "ambiguous" and u.field.label and u.field.input_type != "file"
    ]

    if ambiguous:
        # For combobox-style fields, peek at their actual valid options
        # BEFORE asking the LLM, so it picks one exactly instead of
        # writing free text that then has to be pattern-matched after
        # the fact — a real run showed the LLM writing full sentences
        # that failed to match rendered "Yes"/"No" options at all.
        combobox_options: dict[str, list[str]] = {}
        too_many_options = []
        for u in ambiguous:
            role = await page.locator(f"#{u.field.element_id}").get_attribute("role")
            if role == "combobox":
                options = await get_combobox_options(page, u.field.element_id)
                if not options:
                    continue
                if len(options) > _MAX_COMBOBOX_OPTIONS_FOR_LLM:
                    too_many_options.append(u)
                else:
                    combobox_options[u.field.element_id] = options

        for u in too_many_options:
            fields_skipped.append(
                FieldOutcome(
                    u.field.element_id, u.field.label, False,
                    "too many dropdown options to reliably auto-select", source="skipped",
                )
            )
        ambiguous = [u for u in ambiguous if u not in too_many_options]

        llm = ChatLiteLLM(model=MODEL_NAME, api_base=OLLAMA_BASE_URL)
        job_context = await page.title()
        answers = await answer_unmatched_fields(
            llm, ambiguous, profile, job_context, options_by_id=combobox_options
        )

        for u in ambiguous:
            answer = answers.get(u.field.element_id)
            if not answer:
                continue
            try:
                if u.field.element_id in combobox_options:
                    result = await fill_combobox(page, u.field.element_id, answer)
                    fields_filled.append(
                        FieldOutcome(u.field.element_id, u.field.label, result.ok, result.detail, source="llm")
                    )
                else:
                    await page.locator(f"#{u.field.element_id}").fill(answer)
                    fields_filled.append(
                        FieldOutcome(u.field.element_id, u.field.label, True, answer[:80], source="llm")
                    )
            except Exception as e:
                fields_filled.append(
                    FieldOutcome(u.field.element_id, u.field.label, False, f"could not fill: {e}", source="llm")
                )

    reason = ""
    if resume_missing:
        reason = f"Warning: resume_path '{resume_path}' does not exist — file upload fields were skipped."

    return ApplyResult(
        status=ApplyStatus.READY_FOR_REVIEW,
        url=url,
        reason=reason,
        fields_filled=fields_filled,
        fields_skipped=fields_skipped,
    )


def _print_result(result: ApplyResult) -> None:
    if result.status != ApplyStatus.READY_FOR_REVIEW:
        print(f"\n{result.status.value.upper()}: {result.reason}")
        return

    if result.reason:
        print(result.reason)

    print(f"\n{result.ok_count} field(s) filled successfully:")
    for f in result.fields_filled:
        status = "OK" if f.ok else "FLAGGED"
        print(f"  [{status}] {f.label!r} ({f.source}): {f.detail}")

    if result.fields_skipped:
        print(f"\n{len(result.fields_skipped)} field(s) skipped, needs manual review:")
        for f in result.fields_skipped:
            print(f"  [SKIPPED] {f.label!r}: {f.detail}")

    print("\n--- Form filling complete. Ready for human review. ---")
    print("Jarvis will NOT click Submit/Apply — review the browser window and submit yourself.")


async def run(url: str) -> None:
    """CLI entry point: opens a browser, fills the form, prints the
    result, waits for the human to review, then closes. This is one
    possible caller of fill_application() — n8n or a dashboard would
    call it differently (no input(), likely no headed browser)."""
    browser: Browser | None = None
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            result = await fill_application(page, url)
            _print_result(result)

            if result.status == ApplyStatus.READY_FOR_REVIEW:
                input("\nPress Enter to close the browser...")
        finally:
            if browser is not None:
                await browser.close()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m jarvis.agent <job_posting_url>")
        sys.exit(1)

    asyncio.run(run(sys.argv[1]))


if __name__ == "__main__":
    main()
