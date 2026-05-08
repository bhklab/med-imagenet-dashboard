"""Thin HTTP client for talking to the ADK dev server."""

from __future__ import annotations

import os

import requests

ADK_BASE_URL = os.environ.get("ADK_BASE_URL", "http://localhost:8000")
APP_NAME = "voxel"
USER_ID = "streamlit_user"


def create_session() -> str:
    """Create a new ADK session and return its id."""
    resp = requests.post(
        f"{ADK_BASE_URL}/apps/{APP_NAME}/users/{USER_ID}/sessions",
        json={},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def send_message(session_id: str, message: str) -> str:
    """Send a user message to the agent and return the assistant reply."""
    payload = {
        "app_name": APP_NAME,
        "user_id": USER_ID,
        "session_id": session_id,
        "new_message": {
            "role": "user",
            "parts": [{"text": message}],
        },
        "streaming": False,
    }
    resp = requests.post(
        f"{ADK_BASE_URL}/run",
        json=payload,
        timeout=120,
    )
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(
            f"ADK server returned {resp.status_code}: {detail}"
        )

    events = resp.json()
    agent_text_parts: list[str] = []
    for event in events:
        content = event.get("content")
        if not content or not content.get("parts"):
            continue
        if content.get("role") != "model":
            continue
        for part in content["parts"]:
            if "text" in part:
                agent_text_parts.append(part["text"])

    return "\n".join(agent_text_parts) if agent_text_parts else ""
