"""
Part 1: open a job posting URL and report what form fields are on the page.

No filling, no submitting, no LLM calls — this just proves the browser
automation plumbing (Playwright + browser-use) works before anything is
built on top of it.

Usage:
    python -m jarvis.inspect_form <job_posting_url>
"""

import asyncio
import sys

from browser_use.browser import BrowserProfile, BrowserSession


async def inspect_form(url: str) -> None:
    profile = BrowserProfile(headless=False)
    session = BrowserSession(browser_profile=profile)

    await session.start()
    try:
        await session.navigate_to(url)
        await asyncio.sleep(2)  # let the page settle before reading the DOM

        state = await session.get_browser_state_summary(include_screenshot=False)

        print(f"\nPage title: {state.title}")
        print(f"URL: {state.url}")
        print(f"\nFound {len(state.dom_state.selector_map)} interactive elements:\n")

        for index, element in state.dom_state.selector_map.items():
            tag = element.tag_name
            attrs = element.attributes or {}
            label = (
                attrs.get("aria-label")
                or attrs.get("placeholder")
                or attrs.get("name")
                or element.get_meaningful_text_for_llm()
            )
            field_type = attrs.get("type", "")
            required = "required" in attrs
            print(f"  [{index}] <{tag}{' type=' + field_type if field_type else ''}> "
                  f"{label!r}{' (required)' if required else ''}")

        input("\nPress Enter to close the browser...")
    finally:
        await session.kill()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m jarvis.inspect_form <job_posting_url>")
        sys.exit(1)

    asyncio.run(inspect_form(sys.argv[1]))


if __name__ == "__main__":
    main()
