"""Tests for deterministic routing and agent capability policy."""

from dataclasses import dataclass

from anki_pipeline.routing import KNOWLEDGE_MANAGER_TOOLS, selector_func
from anki_pipeline.tools import fetch_siyuan_notes, push_cards_batch


@dataclass
class Message:
    source: str
    content: str


def test_knowledge_manager_has_read_tool_but_no_anki_write_tool() -> None:
    assert fetch_siyuan_notes in KNOWLEDGE_MANAGER_TOOLS
    assert push_cards_batch not in KNOWLEDGE_MANAGER_TOOLS


def test_reviewer_requires_exact_approval_protocol() -> None:
    assert selector_func([Message("Card_Reviewer", "APPROVED")]) == "Admin"
    assert selector_func([Message("Card_Reviewer", "NOT APPROVED")]) == "Card_Reviewer"


def test_admin_requires_exact_approval_protocol() -> None:
    assert selector_func([Message("Admin", "APPROVE")]) == "Knowledge_Manager"
    assert selector_func([Message("Admin", "NOT APPROVED")]) == "Admin"


def test_explicit_admin_rejection_returns_to_writer() -> None:
    assert selector_func([Message("Admin", "REJECT")]) == "Card_Writer"


def test_two_explicit_reviewer_rejections_escalate_to_human() -> None:
    messages = [
        Message("Card_Reviewer", "REJECTED:\nFirst issue"),
        Message("Card_Writer", '{"cards": []}'),
        Message("Card_Reviewer", "REJECTED:\nSecond issue"),
    ]

    assert selector_func(messages) == "Admin"
