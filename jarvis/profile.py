"""Loads the applicant profile Jarvis uses to fill forms.

Each user supplies their own config/profile.json (gitignored, never
committed) — see config/profile.example.json for the expected shape.
"""

import json
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parent.parent / "config" / "profile.json"
EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "config" / "profile.example.json"


def load_profile() -> dict:
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"No profile found at {PROFILE_PATH}.\n"
            f"Copy {EXAMPLE_PATH} to {PROFILE_PATH} and fill in your own details."
        )
    return json.loads(PROFILE_PATH.read_text())
