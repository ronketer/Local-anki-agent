"""Agent definitions and team orchestration.

This module defines the AutoGen participants. Deterministic routing and tool
capability policy live separately in ``routing.py`` so they can be tested
without starting an LLM runtime.

Architecture:
    User -> Knowledge_Manager -> Card_Writer -> Card_Reviewer -> Admin -> [loop or terminate]

Agentic Design Patterns Used:
    1. Tool Use: The Knowledge Manager can read from Siyuan
    2. Reflection: Card_Reviewer critiques the Card_Writer's output
    3. Multi-Agent Collaboration: Specialized agents with distinct roles
    4. Human-in-the-Loop: Admin provides final approval before saving
"""

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import ChatAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .config import config
from .routing import KNOWLEDGE_MANAGER_TOOLS, selector_func

# System prompts following best practices for instruction clarity
KNOWLEDGE_MANAGER_PROMPT = """You orchestrate a flashcard creation pipeline.

IF content is already provided in the message:
- Simply acknowledge and let Card_Writer create the flashcards.

IF you need to fetch content:
- Use the fetch_siyuan_notes tool with the block ID.

AFTER Admin says exactly 'APPROVE':
- Acknowledge the approval and say TERMINATE.
- Do not attempt to save cards yourself.

IMPORTANT:
- You may fetch source notes, but you never have permission to write to Anki.
- The application performs the final Anki write outside the agent workflow."""

CARD_WRITER_PROMPT = """You are an expert at creating effective Anki flashcards.

Follow the Minimum Information Principle (SuperMemo's 20 Rules):
1. ATOMIC: Each card tests exactly ONE fact
2. CONCISE: Short question, short answer (1-5 words ideal)
3. NO LISTS: Never ask "What are the types of X?" - make separate cards
4. CLEAR: Question should have ONE obvious answer

EXAMPLES:
  Good: Front: "What is a CDN?" Back: "Distributed proxy server network"
  Good: Front: "CDN stands for [...]" Back: "Content Delivery Network"
  Bad:  Front: "What are CDN benefits?" Back: "Faster delivery, reduced server load"

Create 5-10 high-quality cards covering the key concepts.

Output ONLY valid JSON:
{"cards": [{"front": "...", "back": "..."}, ...]}"""

CARD_REVIEWER_PROMPT = """You are a quality reviewer for Anki flashcards.

Quickly check each card:
1. Back has ONE short answer (1-5 words)? PASS
2. Front asks for a list? FAIL
3. Contains markup artifacts like {id=...}? FAIL

If ALL cards pass: Reply with just "APPROVED"
If ANY fail: Reply "REJECTED" then list fixes needed (be brief)"""


def create_model_client() -> OpenAIChatCompletionClient:
    """Create the LLM client configured for the local inference server."""
    # Detect if using Gemini API (for demo with better responses)
    is_gemini = "generativelanguage.googleapis.com" in config.LLM_BASE_URL
    is_ollama = config.LLM_PROVIDER.lower() == "ollama"

    return OpenAIChatCompletionClient(
        model=config.LLM_MODEL_ID,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model_info={
            "json_output": is_gemini,  # Gemini handles JSON well
            "vision": False,
            "function_calling": True,
            "structured_output": is_gemini,
            "family": "unknown",
        },
        # Disable qwen "thinking mode" for faster responses
        extra_create_args={"options": {"num_ctx": 4096}} if is_ollama and not is_gemini else {},
    )


def create_agents(model_client: OpenAIChatCompletionClient) -> dict[str, ChatAgent]:
    """Create all agents for the pipeline."""
    return {
        "knowledge_manager": AssistantAgent(
            name="Knowledge_Manager",
            model_client=model_client,
            tools=list(KNOWLEDGE_MANAGER_TOOLS),
            system_message=KNOWLEDGE_MANAGER_PROMPT,
        ),
        "card_writer": AssistantAgent(
            name="Card_Writer",
            model_client=model_client,
            description="Drafts Anki flashcards from raw notes.",
            system_message=CARD_WRITER_PROMPT,
        ),
        "card_reviewer": AssistantAgent(
            name="Card_Reviewer",
            model_client=model_client,
            description="Critiques flashcards for quality.",
            system_message=CARD_REVIEWER_PROMPT,
        ),
        "admin": UserProxyAgent(
            name="Admin",
            description="Human reviewer who approves flashcards before saving.",
            input_func=lambda prompt: input("\n[APPROVE/REJECT] >>> "),
        ),
    }
