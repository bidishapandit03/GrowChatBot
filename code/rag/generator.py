"""Call the Mistral API. The API key is read from the environment, never from data/ or source."""

from __future__ import annotations

import json
import os

import requests
from dotenv import load_dotenv

from code.config import (
    MISTRAL_API_KEY_ENV,
    MISTRAL_API_URL,
    MISTRAL_GENERATION_TIMEOUT_SECONDS,
    MISTRAL_MAX_TOKENS,
    MISTRAL_MODEL,
    MISTRAL_TEMPERATURE,
)

load_dotenv()


class GenerationError(RuntimeError):
    """Raised when the API key is missing or Mistral fails to return valid JSON."""


def get_api_key() -> str | None:
    """Return the Mistral API key, preferring the environment over a local .env file."""
    return os.environ.get(MISTRAL_API_KEY_ENV) or None


def strip_json_fences(content: str) -> str:
    """Remove common code fences that some models wrap around JSON output."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def generate(messages: list[dict[str, str]]) -> dict:
    """Send a grounded prompt and return the parsed JSON object.

    Raises GenerationError for missing keys, transport failures, HTTP errors, or
    unparseable JSON, so callers can fail closed.
    """
    api_key = get_api_key()
    if not api_key:
        raise GenerationError(f"{MISTRAL_API_KEY_ENV} is not set.")

    payload = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "temperature": MISTRAL_TEMPERATURE,
        "max_tokens": MISTRAL_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=MISTRAL_GENERATION_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise GenerationError(f"Mistral request failed: {exc}") from exc

    if response.status_code != 200:
        raise GenerationError(f"Mistral API returned HTTP {response.status_code}.")

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise GenerationError("Mistral response did not contain a chat completion.") from exc

    try:
        parsed = json.loads(strip_json_fences(content))
    except (json.JSONDecodeError, TypeError) as exc:
        raise GenerationError("Mistral response was not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise GenerationError("Mistral returned a non-object JSON value.")
    return parsed