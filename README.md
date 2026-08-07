# Jarvis — Autonomous Job Application Agent

Jarvis is an open-source, self-hosted agent that navigates real job application forms, fills them out, and **always stops for your approval before submitting**. It runs entirely on your own machine, using a local LLM — no API keys, no cloud costs, no subscriptions.

It's built to be transparent about what it will and won't do:

- It fills forms and answers short custom questions ("why this company?") on your behalf.
- It **never** submits an application without you explicitly approving it.
- It **never** attempts to bypass CAPTCHAs or anti-bot protections — that's a deliberate hard stop, not a missing feature. When Jarvis hits one, it flags the application for you instead.
- It can check your inbox for OTP/verification codes and track application outcomes (rejections, interview invites) — using your own Gmail account, with your own OAuth consent.

## How it works

```
Load next opening (n8n)
        │
        ▼
Jarvis fills the form  ──┐   perceive → decide → act loop
 (browser-use + local LLM)│   powered by Qwen, served by Ollama
        │                 │
        ▼                 │
   Stuck / CAPTCHA? ───────┘── flagged → logged, shown in dashboard
        │ no
        ▼
  Waits for your approval (dashboard)
        │ approved
        ▼
  Jarvis submits — and only this one action
        │
        ▼
  Gmail monitoring tracks the outcome (OTP, rejection, interview)
```

Jarvis's own authority is intentionally narrow: it decides *how* to fill a form, never *whether* to submit it, and never *how* to defeat a security control.

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| LLM | [Qwen3](https://github.com/QwenLM/Qwen) (8B) | Open-weight, strong tool-use/reasoning at a size that runs locally |
| LLM runtime | [Ollama](https://ollama.com) | Serves Qwen locally via an OpenAI-compatible API |
| Browser agent | [browser-use](https://github.com/browser-use/browser-use) + Playwright | DOM-aware perceive→decide→act loop for real webpages |
| Orchestration | [n8n](https://n8n.io) (self-hosted) | Loads openings, routes flagged/approved/submit states, schedules Gmail polling |
| Email | Gmail API | OTP retrieval and application status tracking, per-user OAuth |
| Dashboard | Local backend + frontend, exposed via Cloudflare Tunnel | Interactive tracking and approval, self-hosted like everything else |

Everything runs locally and is free to use — no paid APIs required.

## Status

This project is under active development. Nothing is wired up yet — see the roadmap below.

- [ ] Core browser-use loop (open a form, read its fields)
- [ ] LLM decision loop (Ollama + Qwen)
- [ ] Approval gate (fill → stop before submit → structured result)
- [ ] n8n orchestration pipeline
- [ ] Gmail monitoring (OTP + status tracking)
- [ ] Interactive dashboard
- [ ] One-command setup (Docker Compose) for self-hosting

## Setup

Setup instructions will land here once the core pipeline works end-to-end. The goal is a self-hostable stack anyone can run with their own resume, job list, and Gmail account — no shared credentials, no hosted service.


## License

MIT — see [LICENSE](LICENSE).
