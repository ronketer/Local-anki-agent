"""Deterministic agent routing and capability policy.

This module intentionally has no AutoGen dependency. The routing protocol and
agent tool capabilities can therefore be unit-tested without starting an LLM
client or conversation runtime.
"""

from __future__ import annotations

from typing import Protocol

from .tools import fetch_siyuan_notes
from .workflow import HumanDecision, ReviewDecision, parse_human_decision, parse_review_decision

KNOWLEDGE_MANAGER_TOOLS = (fetch_siyuan_notes,)


class RoutingMessage(Protocol):
    """Minimal message shape used by the deterministic selector."""

    source: str
    content: object


def selector_func(messages: list[RoutingMessage]) -> str | None:
    """Choose the next participant using exact, deterministic protocol rules."""
    if not messages or messages[-1].source == "user":
        return "Knowledge_Manager"

    last = messages[-1]

    if last.source == "Knowledge_Manager":
        return "Card_Writer"

    if last.source == "Card_Writer":
        return "Card_Reviewer"

    if last.source == "Card_Reviewer":
        decision = parse_review_decision(str(last.content))
        if decision == ReviewDecision.APPROVED:
            return "Admin"

        if decision == ReviewDecision.REJECTED:
            rejection_count = sum(
                1
                for message in messages
                if getattr(message, "source", None) == "Card_Reviewer"
                and parse_review_decision(str(getattr(message, "content", "")))
                == ReviewDecision.REJECTED
            )
            if rejection_count >= 2:
                return "Admin"
            return "Card_Writer"

        # Invalid reviewer output is not interpreted as a decision.
        return "Card_Reviewer"

    if last.source == "Admin":
        decision = parse_human_decision(str(last.content))
        if decision == HumanDecision.APPROVED:
            return "Knowledge_Manager"
        if decision == HumanDecision.REJECTED:
            return "Card_Writer"

        # Invalid human input is re-prompted rather than guessed.
        return "Admin"

    return None
