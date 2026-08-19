"""Application-owned orchestration helpers.

AutoGen provides the conversation transport, but authorization is derived by
replaying agent messages through the deterministic ``PipelineRun`` state
machine. No LLM message can directly authorize an external Anki write.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import ValidationError

from .models import FlashcardList
from .workflow import (
    HumanDecision,
    PipelineRun,
    ReviewDecision,
    WorkflowStage,
    parse_human_decision,
    parse_review_decision,
)


class WorkflowMessage(Protocol):
    """Minimal message shape required to replay a conversation."""

    source: str
    content: object


def extract_flashcards(content: str) -> FlashcardList | None:
    """Extract and validate a flashcard list from an agent message."""
    patterns = (
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r'(\{"cards":\s*\[.*?\]\})',
    )

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            continue

        try:
            payload = json.loads(match.group(1))
            return FlashcardList.model_validate(payload)
        except (json.JSONDecodeError, ValidationError):
            continue

    return None


def replay_workflow(block_id: str, messages: list[WorkflowMessage]) -> PipelineRun:
    """Build authoritative workflow state from a completed agent transcript.

    Conversation text is treated as untrusted input. Only validated card JSON
    and exact protocol decisions are allowed to advance the state machine.
    """
    run = PipelineRun(block_id=block_id)
    run.content_ready()

    for message in messages:
        source = getattr(message, "source", "")
        content = str(getattr(message, "content", ""))

        if source == "Card_Writer" and run.stage == WorkflowStage.GENERATING:
            cards = extract_flashcards(content)
            if cards is not None and cards.cards:
                run.draft_ready(cards)
            continue

        if source == "Card_Reviewer" and run.stage == WorkflowStage.REVIEWING:
            decision = parse_review_decision(content)
            if decision == ReviewDecision.APPROVED:
                run.reviewer_approved()
            elif decision == ReviewDecision.REJECTED:
                run.reviewer_rejected()
            continue

        if source == "Admin" and run.stage == WorkflowStage.AWAITING_HUMAN:
            decision = parse_human_decision(content)
            if decision == HumanDecision.APPROVED:
                run.human_approved()
            elif decision == HumanDecision.REJECTED:
                run.human_rejected()

    return run
