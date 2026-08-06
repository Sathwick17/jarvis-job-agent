"""
Part 2: Jarvis's actual decision loop.

Given a job posting URL and the applicant's profile, Jarvis looks at the
page, decides what to fill in, and acts — repeating until the form is
filled or it gets stuck. It never submits; that's a hard stop enforced
by the task instructions given to the agent (Part 3 will make this a
structured, code-enforced boundary rather than just a prompted one).

Uses Qwen, served locally by Ollama — no API keys, no cloud calls.

Usage:
    python -m jarvis.agent <job_posting_url>
"""

import asyncio
import os
import sys
from pathlib import Path

from browser_use import Agent
from browser_use.browser import BrowserProfile
from browser_use.llm.litellm.chat import ChatLiteLLM

from jarvis.profile import load_profile

OLLAMA_BASE_URL = "http://localhost:11434"
# qwen2.5:3b is faster but unreliable — it hallucinated an unrelated task
# (tried to print the page as a PDF) instead of following instructions.
# qwen3:8b is slower (3+ min/step locally) but actually stays on-task.
MODEL_NAME = os.environ.get("JARVIS_MODEL", "ollama/qwen3:8b")


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
1. Navigate to the application form for this posting (click "Apply" if you're on a
   job description page first).
2. Fill in every visible field using the applicant details above. If a field asks
   something you don't have data for (e.g. a custom question), give a brief,
   reasonable answer based on the job context.
3. If you encounter a CAPTCHA, a login requirement, or anything you cannot
   confidently complete, STOP immediately and report what blocked you. Do not
   attempt to bypass or work around it.
4. Once the form is completely filled in, STOP. Do NOT click Submit, Apply, or
   any final-submission button under any circumstances. Report that the form is
   ready for human review.

Important: waiting for a page to load, or merely observing that fields exist, is
NOT progress and is NOT a reason to mark the task done. You are only done once
every visible field has actually been filled in with real values (or you are
truly blocked by something in rule 3). If a page looks empty, wait briefly and
try again before giving up — do not report success until you have taken real
fill-in actions.

Stay strictly within these instructions. Do not print, download, save, or export
the page in any form (no Ctrl+P, no "save as PDF", no file exports) — that is
never part of this task. Do not research or evaluate which job to apply for. Do
not use the resume file for anything other than uploading it to a resume/CV
upload field on the form itself. If you are unsure what to do next, re-read
these instructions rather than inventing a new goal.
"""


async def run(url: str) -> None:
    profile = load_profile()
    task = build_task(url, profile)

    llm = ChatLiteLLM(model=MODEL_NAME, api_base=OLLAMA_BASE_URL)
    browser_profile = BrowserProfile(headless=False)

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
