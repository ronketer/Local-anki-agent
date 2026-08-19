"""External-system adapters used by the pipeline.

Adapters return successful data and raise typed exceptions on failure. They do
not implement retry policy themselves; callers decide whether an operation is
safe to repeat.
"""

from __future__ import annotations

import json
import re
from typing import Annotated

import requests

from .config import config
from .errors import (
    AnkiResponseError,
    AnkiUnavailableError,
    PayloadValidationError,
    SiyuanResponseError,
    SiyuanUnavailableError,
)


def _clean_kramdown(kramdown: str) -> str:
    """Remove Siyuan metadata from kramdown for cleaner display."""
    cleaned = re.sub(r'\{:\s*id="[^"]+"[^}]*\}', "", kramdown)
    cleaned = re.sub(r'\{:\s*updated="[^"]+"[^}]*\}', "", cleaned)
    cleaned = re.sub(r"\{\{\{row\n?", "", cleaned)
    cleaned = re.sub(r"\}\}\}", "", cleaned)
    cleaned = re.sub(r"\s*\{:\s*[^}]+\}", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def fetch_siyuan_notes(
    block_id: Annotated[str, "The unique 22-character Siyuan block ID to fetch."],
) -> str:
    """Fetch and return cleaned Siyuan block data as JSON.

    Temporary connectivity/server failures raise ``SiyuanUnavailableError``.
    Request rejection or malformed responses raise ``SiyuanResponseError``.
    """
    headers = (
        {"Authorization": f"Token {config.SIYUAN_API_TOKEN}"}
        if config.SIYUAN_API_TOKEN
        else {}
    )
    payload = {"id": block_id}

    try:
        response = requests.post(
            config.SIYUAN_API_URL,
            headers=headers,
            json=payload,
            timeout=10,
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        raise SiyuanUnavailableError(
            "Cannot connect to Siyuan or the request timed out"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise SiyuanResponseError(f"Siyuan request failed: {exc}") from exc

    if response.status_code >= 500:
        raise SiyuanUnavailableError(
            f"Siyuan returned temporary HTTP {response.status_code}"
        )
    if response.status_code != 200:
        raise SiyuanResponseError(f"Siyuan returned HTTP {response.status_code}")

    try:
        response_data = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise SiyuanResponseError("Siyuan returned invalid JSON") from exc

    if not isinstance(response_data, dict):
        raise SiyuanResponseError("Siyuan returned an unexpected response shape")

    if response_data.get("code") != 0:
        message = response_data.get("msg") or "request rejected"
        raise SiyuanResponseError(f"Siyuan rejected the request: {message}")

    data = response_data.get("data")
    if not isinstance(data, dict):
        raise SiyuanResponseError("Siyuan response is missing an object 'data' field")

    if "kramdown" in data:
        if not isinstance(data["kramdown"], str):
            raise SiyuanResponseError("Siyuan 'kramdown' field must be a string")
        data = dict(data)
        data["kramdown"] = _clean_kramdown(data["kramdown"])

    return json.dumps(data, indent=2, ensure_ascii=False)


def _push_to_anki(
    front_text: Annotated[str, "The text for the front of the flashcard."],
    back_text: Annotated[str, "The text for the back of the flashcard."],
) -> str:
    """Push a single flashcard to Anki via the AnkiConnect API."""
    payload = {
        "action": "addNote",
        "version": 6,
        "params": {
            "note": {
                "deckName": config.ANKI_DECK_NAME,
                "modelName": "Basic",
                "fields": {"Front": front_text, "Back": back_text},
                "options": {"allowDuplicate": False},
            }
        },
    }

    try:
        response = requests.post(config.ANKI_CONNECT_URL, json=payload, timeout=10)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        raise AnkiUnavailableError(
            "Cannot connect to AnkiConnect or the request timed out"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise AnkiResponseError(f"AnkiConnect request failed: {exc}") from exc

    if response.status_code >= 500:
        raise AnkiUnavailableError(
            f"AnkiConnect returned temporary HTTP {response.status_code}"
        )
    if response.status_code != 200:
        raise AnkiResponseError(f"AnkiConnect returned HTTP {response.status_code}")

    try:
        response_data = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise AnkiResponseError("AnkiConnect returned invalid JSON") from exc

    if not isinstance(response_data, dict):
        raise AnkiResponseError("AnkiConnect returned an unexpected response shape")

    error = response_data.get("error")
    if error is not None:
        raise AnkiResponseError(f"AnkiConnect rejected the note: {error}")

    if response_data.get("result") is None:
        raise AnkiResponseError("AnkiConnect response is missing a note id")

    return f"Card added: {response_data['result']}"


def push_cards_batch(
    cards_json: Annotated[
        str,
        'JSON string: {"cards": [{"front": "...", "back": "..."}]}',
    ],
) -> str:
    """Push multiple validated flashcards to Anki.

    This adapter intentionally does not retry writes. Until the write path is
    idempotent, repeating a partially completed batch could create duplicates.
    """
    try:
        data = json.loads(cards_json)
    except json.JSONDecodeError as exc:
        raise PayloadValidationError("Card batch is not valid JSON") from exc

    cards = data.get("cards") if isinstance(data, dict) else None
    if not isinstance(cards, list) or not cards:
        raise PayloadValidationError("Card batch must contain at least one card")

    validated_cards: list[tuple[str, str]] = []
    for index, card in enumerate(cards, 1):
        if not isinstance(card, dict):
            raise PayloadValidationError(f"Card {index} must be an object")

        front = card.get("front", card.get("question", ""))
        back = card.get("back", card.get("answer", ""))
        if not isinstance(front, str) or not front.strip():
            raise PayloadValidationError(f"Card {index} is missing a non-empty front")
        if not isinstance(back, str) or not back.strip():
            raise PayloadValidationError(f"Card {index} is missing a non-empty back")

        validated_cards.append((front.strip(), back.strip()))

    # Validate the complete batch before the first external side effect.
    results: list[str] = []
    for index, (front, back) in enumerate(validated_cards, 1):
        result = _push_to_anki(front, back)
        results.append(f"Card {index}: {result}")

    return f"Saved {len(results)} cards:\n" + "\n".join(results)
