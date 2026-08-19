"""Offline tests for Siyuan and Anki HTTP adapters."""

import json

import pytest
import requests

from anki_pipeline.errors import (
    AnkiResponseError,
    AnkiUnavailableError,
    PayloadValidationError,
    SiyuanResponseError,
    SiyuanUnavailableError,
)
from anki_pipeline.tools import (
    _push_to_anki,
    add_note,
    fetch_siyuan_notes,
    find_note_ids_by_tag,
    push_cards_batch,
)


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: object | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error

    def json(self) -> object:
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def test_fetch_siyuan_notes_returns_cleaned_data(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(
        payload={
            "code": 0,
            "data": {
                "kramdown": 'Atomic fact {: id="20260101000000-test"}'
            },
        }
    )
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)

    result = json.loads(fetch_siyuan_notes("20260101000000-test"))

    assert result["kramdown"] == "Atomic fact"


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.ConnectionError("refused"),
        requests.exceptions.Timeout("slow"),
    ],
)
def test_siyuan_connectivity_failure_is_transient(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(requests, "post", fail)

    with pytest.raises(SiyuanUnavailableError):
        fetch_siyuan_notes("20260101000000-test")


def test_siyuan_server_error_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(status_code=503, payload={}),
    )

    with pytest.raises(SiyuanUnavailableError, match="HTTP 503"):
        fetch_siyuan_notes("20260101000000-test")


def test_siyuan_request_rejection_is_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            payload={"code": -1, "msg": "block not found"}
        ),
    )

    with pytest.raises(SiyuanResponseError, match="block not found"):
        fetch_siyuan_notes("missing")


def test_anki_connectivity_failure_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fail)

    with pytest.raises(AnkiUnavailableError):
        _push_to_anki("Q?", "A")


def test_anki_rejection_is_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            payload={"result": None, "error": "cannot create note"}
        ),
    )

    with pytest.raises(AnkiResponseError, match="cannot create note"):
        _push_to_anki("Q?", "A")


def test_batch_rejects_invalid_json_before_any_write() -> None:
    with pytest.raises(PayloadValidationError, match="not valid JSON"):
        push_cards_batch("{not-json")


def test_batch_rejects_missing_card_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def should_not_call(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("HTTP should not be called")

    monkeypatch.setattr(requests, "post", should_not_call)

    with pytest.raises(PayloadValidationError, match="missing a non-empty back"):
        push_cards_batch('{"cards": [{"front": "Q?", "back": ""}]}')

    assert calls == 0


def test_entire_batch_is_validated_before_first_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def should_not_call(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("HTTP should not be called")

    monkeypatch.setattr(requests, "post", should_not_call)

    with pytest.raises(PayloadValidationError, match="Card 2"):
        push_cards_batch(
            '{"cards": ['
            '{"front": "valid", "back": "valid"},'
            '{"front": "invalid", "back": ""}'
            ']}'
        )

    assert calls == 0


def test_find_note_ids_by_tag_uses_anki_search_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return FakeResponse(payload={"result": [101, 202], "error": None})

    monkeypatch.setattr(requests, "post", fake_post)

    result = find_note_ids_by_tag("local_anki_agent_id_abc123")

    assert result == [101, 202]
    assert captured["json"] == {
        "action": "findNotes",
        "version": 6,
        "params": {"query": "tag:local_anki_agent_id_abc123"},
    }


def test_find_note_ids_by_tag_rejects_invalid_result_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            payload={"result": ["not-a-note-id"], "error": None}
        ),
    )

    with pytest.raises(AnkiResponseError, match="invalid note ids"):
        find_note_ids_by_tag("local_anki_agent_id_abc123")


def test_add_note_sends_idempotency_tag_and_returns_note_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["json"] = kwargs["json"]
        return FakeResponse(payload={"result": 987654321, "error": None})

    monkeypatch.setattr(requests, "post", fake_post)

    note_id = add_note(
        "What is idempotency?",
        "Safe repetition",
        ["local_anki_agent_id_abc123"],
    )

    assert note_id == 987654321
    assert captured["json"] == {
        "action": "addNote",
        "version": 6,
        "params": {
            "note": {
                "deckName": "Default",
                "modelName": "Basic",
                "fields": {
                    "Front": "What is idempotency?",
                    "Back": "Safe repetition",
                },
                "tags": ["local_anki_agent_id_abc123"],
                "options": {"allowDuplicate": False},
            }
        },
    }
