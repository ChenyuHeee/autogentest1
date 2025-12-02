"""Factory for the human-in-the-loop admin proxy."""

from __future__ import annotations

from autogen import UserProxyAgent


def create_admin_proxy() -> UserProxyAgent:
    """Return a user proxy that always prompts a human operator."""

    return UserProxyAgent(
        name="AdminUserProxy",
        system_message=(
            "You are the escalation admin. When prompted, decide whether to force execution, stand down, "
            "or provide custom instructions. Respond with concise JSON: {\"decision\": <FORCE_EXECUTE|STAND_DOWN|CUSTOM>, \"notes\": "
            "<optional context>}"
        ),
        human_input_mode="ALWAYS",
        code_execution_config={"use_docker": False},
    )
