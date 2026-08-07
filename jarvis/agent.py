"""
Part 2: Jarvis's form-filling pipeline.

Deterministic-first: reads the real form fields via Playwright, fills
whatever matches the applicant's profile in code (no LLM), and only
calls the LLM for the handful of fields that are genuinely ambiguous
(custom free-text questions). This replaced an earlier full-agent-loop
design where an LLM re-interpreted the entire page on every step — that
approach was slow (3+ min/step locally), and in testing hallucinated
fake URLs, got stuck in scroll loops, and once clicked a real "Submit
application" button despite explicit instructions not to.

Jarvis never submits: this pipeline has no code path that clicks
anything after fields are filled — it stops and waits for a human to
review and submit manually. That's a stronger guarantee than a runtime
check (see jarvis/safe_tools.py, still used by the older LLM-loop
approach) because there's simply no click-executing code left to guard.
The only click this pipeline ever performs is "Apply" on a listing
page, before any filling happens.

Uses Qwen, served locally by Ollama — no API keys, no cloud calls.

Usage:
    python -m jarvis.agent <job_posting_url>
"""

import asyncio
import sys
from pathlib import Path

from playwright.async_api import Page, async_playwright

from browser_use.llm.litellm.chat import ChatLiteLLM

from jarvis.field_matcher import match_fields
from jarvis.form_filler import fill_combobox, fill_matched_fields
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


async def run(url: str) -> None:
    profile = load_profile()

    resume_path = Path(profile.get("resume_path", ""))
    if profile.get("resume_path") and not resume_path.exists():
        print(f"Warning: resume_path '{resume_path}' does not exist — file upload fields will be skipped.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        print(f"Navigating to {url} ...")
        await page.goto(url, timeout=30000)
        await page.wait_for_timeout(_PAGE_LOAD_WAIT_MS)

        fields = await read_form_fields(page)
        if len(fields) < 3:
            # Likely a job listing page, not the application form itself.
            print("Few/no form fields found — looking for an Apply link...")
            clicked = await _click_apply_if_present(page)
            if clicked:
                fields = await read_form_fields(page)
            if len(fields) < 3:
                print(f"Still only found {len(fields)} field(s). Stopping — "
                      "this may not be a direct application form URL.")
                await browser.close()
                return

        print(f"Found {len(fields)} form fields.")

        matched, unmatched = match_fields(fields, profile)
        captcha_fields = [u for u in unmatched if u.reason == "captcha"]
        if captcha_fields:
            print("\nCAPTCHA detected on this form. Stopping — Jarvis does not "
                  "attempt to solve or bypass CAPTCHAs by design.")
            await browser.close()
            return

        print(f"\nDeterministically matched {len(matched)} fields (no LLM):")
        fill_results = await fill_matched_fields(page, matched)
        for r in fill_results:
            status = "OK" if r.ok else "FLAGGED"
            print(f"  [{status}] {r.label!r}: {r.detail}")

        # File inputs (e.g. an optional cover-letter upload we have no file
        # for) are never something the LLM can "answer" with text — a real
        # run tried exactly this and Playwright correctly rejected it
        # (file inputs can't be .fill()'d). Report them as skipped instead.
        file_inputs = [u for u in unmatched if u.reason == "ambiguous" and u.field.input_type == "file"]
        ambiguous = [
            u for u in unmatched
            if u.reason == "ambiguous" and u.field.label and u.field.input_type != "file"
        ]
        if file_inputs:
            print(f"\nSkipped {len(file_inputs)} optional file upload(s) with no matching file "
                  f"(e.g. cover letter): {[u.field.label for u in file_inputs]}")
        skipped_demographic = [u for u in unmatched if u.reason == "skipped_demographic"]
        if skipped_demographic:
            print(f"\nSkipped {len(skipped_demographic)} demographic/EEO field(s) — "
                  "never auto-answered, applicant's own choice.")

        if ambiguous:
            print(f"\nAsking the LLM to answer {len(ambiguous)} ambiguous field(s) "
                  f"(one focused call per field, not the whole page)...")
            llm = ChatLiteLLM(model=MODEL_NAME, api_base=OLLAMA_BASE_URL)
            job_context = await page.title()
            answers = await answer_unmatched_fields(llm, ambiguous, profile, job_context)

            for u in ambiguous:
                answer = answers.get(u.field.element_id)
                if not answer:
                    continue
                locator = page.locator(f"#{u.field.element_id}")
                try:
                    role = await locator.get_attribute("role")
                    if role == "combobox":
                        # Most Yes/No-style custom questions are the same
                        # React-Select widget as Country — a plain fill()
                        # doesn't register as a real selection there either.
                        result = await fill_combobox(page, u.field.element_id, answer)
                        status = "OK" if result.ok else "FLAGGED"
                        print(f"  [{status}] {u.field.label!r}: {result.detail}")
                    else:
                        await locator.fill(answer)
                        print(f"  [OK] {u.field.label!r}: {answer[:60]!r}")
                except Exception as e:
                    print(f"  [FLAGGED] {u.field.label!r}: could not fill — {e}")

        print("\n--- Form filling complete. Ready for human review. ---")
        print("Jarvis will NOT click Submit/Apply — review the browser window and submit yourself.")
        input("\nPress Enter to close the browser...")
        await browser.close()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m jarvis.agent <job_posting_url>")
        sys.exit(1)

    asyncio.run(run(sys.argv[1]))


if __name__ == "__main__":
    main()
