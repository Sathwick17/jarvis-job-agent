"""
Part 2: Jarvis's actual decision loop.

Given a job posting URL and the applicant's profile, Jarvis looks at the
page, decides what to fill in, and acts — repeating until the form is
filled or it gets stuck. It never submits: this is enforced both by the
task instructions AND, more importantly, at the code level — see
jarvis/safe_tools.py, which physically blocks any click on a submit-like
element regardless of what the model decides.

Uses Qwen, served locally by Ollama — no API keys, no cloud calls.

Usage:
    python -m jarvis.agent <job_posting_url>
"""

import asyncio
import os
import sys
from pathlib import Path

# Must be set before browser_use.browser.events is imported (its Field
# default_factory reads this at BrowserSession/Agent construction time).
# Local inference (Ollama) competing for CPU/GPU can make DOM-tree building
# exceed the default 30s budget, which silently returns an empty page state
# instead of erroring — Jarvis then thinks the page never loaded.
os.environ.setdefault("TIMEOUT_BrowserStateRequestEvent", "90")

from browser_use import Agent
from browser_use.browser import BrowserProfile
from browser_use.llm.litellm.chat import ChatLiteLLM

from jarvis.profile import load_profile
from jarvis.safe_tools import build_safe_tools

OLLAMA_BASE_URL = "http://localhost:11434"
# qwen2.5:3b is faster but unreliable — it hallucinated an unrelated task
# (tried to print the page as a PDF) instead of following instructions.
# qwen3:8b's default 4096-token context was too small (truncation likely
# contributed to hallucinated URLs/goals). qwen3:8b-32k fixed that but its
# KV cache consumed ~7.4GB RAM, leaving the 18GB machine starved enough
# that the model started emitting degenerate repeated-token output.
# qwen3:8b-8k is sized to what this task actually needs (~1-2K tokens of
# instructions + DOM dump) without exhausting system memory.
MODEL_NAME = os.environ.get("JARVIS_MODEL", "ollama/qwen3:8b-8k")


def build_task(url: str, profile: dict) -> str:
    return f"""
You are filling out a job application form at {url}.

Applicant details:
- Full name: {profile['full_name']}
- Email: {profile['email']}
- Phone: {profile['phone']}
- Location: {profile['location']}
- LinkedIn: {profile.get('linkedin_url', '')}
- GitHub: {profile.get('github_url', '')}
- Portfolio: {profile.get('portfolio_url', '')}
- Work authorization: {profile.get('work_authorization', '')}
- Sponsorship: {profile.get('sponsorship_required', '')}
- Resume file (upload this for any resume/CV field): {profile.get('resume_path', '')}
- Years of experience: {profile.get('years_of_experience', '')}

Instructions:
1. Navigate to the application form for this posting. If the URL above is a list
   of multiple job openings rather than one specific job, click on any single
   job listing first to open that job's page — do not scroll looking for an
   "Apply" button on the list page itself, it is not there. Once you are on a
   single job's page, click its "Apply" button/link to reach the actual form.
2. Fill in every visible field using the applicant details above. If a field asks
   something you don't have data for (e.g. a custom question), give a brief,
   reasonable answer based on the job context.
3. If you encounter a CAPTCHA, a login requirement, or anything you cannot
   confidently complete, STOP immediately and report what blocked you. Do not
   attempt to bypass or work around it.
4. Once the form is completely filled in, STOP. Do NOT click Submit, Apply, or
   any final-submission button under any circumstances. Report that the form is
   ready for human review.

Important: waiting for a page to load, merely observing that fields exist, or
figuring out what you should click or fill next, is NOT progress and is NEVER a
reason to call the "done" action. Only call "done" once every visible field has
actually been filled in with real values (or you are truly blocked by something
in rule 3 — a real CAPTCHA/login wall, not just a button that didn't click on
the first try). Explaining what you plan to do next is not the same as doing
it — never call "done" with success=True just because you know the next step;
take that step instead. If a click doesn't seem to have worked, look at the
page again and try a different element (e.g. the text label instead of an icon
inside it) before giving up.

Stay strictly within these instructions. Do not print, download, save, or export
the page in any form (no Ctrl+P, no "save as PDF", no file exports) — that is
never part of this task. Do not research or evaluate which job to apply for. Do
not use the resume file for anything other than uploading it to a resume/CV
upload field on the form itself. If you are unsure what to do next, re-read
these instructions rather than inventing a new goal.

Never type or construct a URL yourself, and never use the navigate action to
guess at an application URL. URLs like job IDs, campaign parameters, or apply
links are not something you can know or infer — any URL you make up will be
wrong. The only way to reach the application form is to click an actual
"Apply" link or button that is visible on the current page. If you cannot find
one, scroll to look for it or go back to the original page above — do not
navigate to any URL that was not explicitly given to you or read directly off
the page you are looking at.
"""


async def run(url: str) -> None:
    profile = load_profile()
    task = build_task(url, profile)

    llm = ChatLiteLLM(model=MODEL_NAME, api_base=OLLAMA_BASE_URL)
    browser_profile = BrowserProfile(
        headless=False,
        # Greenhouse's job board is a JS-heavy React app; the 0.25s/0.5s
        # defaults aren't enough time for it to render before Jarvis reads
        # the page, which was causing it to see an empty page every time.
        minimum_wait_page_load_time=2.0,
        wait_for_network_idle_page_load_time=3.0,
    )

    resume_path = Path(profile.get("resume_path", ""))
    available_file_paths = [str(resume_path.resolve())] if resume_path.exists() else None
    if profile.get("resume_path") and not resume_path.exists():
        print(f"Warning: resume_path '{resume_path}' does not exist — file upload fields will be skipped.")

    agent = Agent(
        task=task,
        llm=llm,
        browser_profile=browser_profile,
        available_file_paths=available_file_paths,
        use_vision=False,  # qwen3:8b via Ollama is text-only; screenshots make every call fail
        llm_timeout=180,  # local 8B inference on a laptop is slower than a hosted API
        tools=build_safe_tools(),  # code-level block on submit-like clicks — see safe_tools.py
        # Default of 5 caused Jarvis to blindly fire 5 actions per LLM call
        # without re-reading the page between them — it typed 5 different
        # values into the same stale element index repeatedly. Forcing it
        # to re-observe more often fixes that at the cost of more steps.
        max_actions_per_step=1,
    )
    history = await agent.run()

    print("\n--- Jarvis run finished ---")
    print(f"Steps taken: {len(history.history)}")
    print(f"Final result: {history.final_result()}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m jarvis.agent <job_posting_url>")
        sys.exit(1)

    asyncio.run(run(sys.argv[1]))


if __name__ == "__main__":
    main()
