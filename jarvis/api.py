"""
Local HTTP API wrapping fill_application() for non-Python callers.

Why an HTTP API and not, say, an orchestrator running Python directly:
this project is meant to be self-hosted by other people, and the
orchestration layer (n8n) is Node.js — an HTTP call is the one
integration path that works the same regardless of what's calling it
(n8n's HTTP Request node today, a dashboard backend in Part 5, or
anything else later) without requiring the caller to know anything
about this codebase's language, venv, or file layout. Same reasoning
as Part 3's ApplyResult contract, one layer up.

Runs entirely on localhost — nothing here is exposed externally unless
you explicitly choose to (e.g. via the Cloudflare Tunnel mentioned in
the README for the dashboard).

Usage:
    uvicorn jarvis.api:app --port 8420
"""

import uuid
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from playwright.async_api import Browser, Page, async_playwright
from pydantic import BaseModel

from jarvis.agent import fill_application
from jarvis.apply_result import ApplyResult

app = FastAPI(title="Jarvis API", description="Local job-application form-filling API")

# Each in-progress/awaiting-review application keeps its own browser page
# open, keyed by a session id, so a caller can fill a form via one request
# and the browser stays visible for human review until they explicitly
# close it via DELETE — mirrors the CLI's "stop and wait for Enter"
# behavior, just over HTTP instead of a blocking input() call.
_sessions: dict[str, tuple[Browser, Page]] = {}


class ApplyRequest(BaseModel):
    url: str


class SessionResult(BaseModel):
    session_id: str
    result: dict


@app.post("/apply", response_model=SessionResult)
async def apply(request: ApplyRequest) -> SessionResult:
    """Fills out a job application form. The browser window opens and
    stays open after this returns — call DELETE /sessions/{session_id}
    once a human has reviewed and (manually) submitted or dismissed it.
    """
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    page = await browser.new_page()

    try:
        result: ApplyResult = await fill_application(page, request.url)
    except Exception as e:
        await browser.close()
        await playwright.stop()
        raise HTTPException(status_code=500, detail=f"fill_application failed: {e}")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = (browser, page)

    return SessionResult(session_id=session_id, result=_result_to_dict(result))


@app.delete("/sessions/{session_id}")
async def close_session(session_id: str) -> dict:
    """Closes the browser for a completed review. Jarvis never calls
    this on its own — closing (and any prior submission) is always a
    decision made outside this codebase, by whoever is orchestrating."""
    session = _sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    browser, _ = session
    await browser.close()
    return {"closed": session_id}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _result_to_dict(result: ApplyResult) -> dict:
    data = asdict(result)
    data["status"] = result.status.value  # Enum -> plain string for JSON
    data["ok_count"] = result.ok_count
    data["needs_attention_count"] = result.needs_attention_count
    return data
