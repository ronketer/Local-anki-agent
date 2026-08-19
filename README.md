# Autonomous Local Knowledge to Anki Pipeline

A **multi-agent AI system** that extracts knowledge from [Siyuan Notes](https://b3log.org/siyuan/) and generates optimized [Anki](https://apps.ankiweb.net/) flashcards using [Microsoft AutoGen](https://microsoft.github.io/autogen/).

> **Privacy-First**: Runs entirely locally using Ollama with GPU acceleration. No data leaves your machine.

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![AutoGen](https://img.shields.io/badge/AutoGen-0.4+-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-0.17+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Architecture Decisions

This pipeline demonstrates **production-grade agentic design patterns** with explicit tradeoffs. All architectural decisions are enforced in code, not just in prompts.

### 1. Code-Driven Routing (Deterministic State Machine)

```
User → Knowledge_Manager → Card_Writer → Card_Reviewer → Admin → [loop or terminate]
```

Agent handoffs use a **`selector_func` state machine** rather than LLM-chosen routing:

- **Why**: Makes control flow testable, auditable, and free of inference cost on orchestration decisions
- **Tradeoff**: Less flexible if you need dynamic agent selection, but vastly simpler debugging
- **Implementation**: `routing.py` contains the routing policy as pure Python with no AutoGen dependency

### 2. Reflection Loop with Rejection Guardrail

The **Card_Reviewer → Card_Writer feedback loop** automatically runs until cards pass quality checks:

- **Why**: Iterative refinement catches obvious failures (lists, verbose answers) without human input
- **Guardrail**: Loop caps at **2 rejections** before escalating to human review
- **Why the cap**: Prevents runaway rewrites and model-specific failure modes (e.g., smaller models stuck in loops)
- **Implementation**: Guardrail enforced in `selector_func`, not just in prompts — a code guarantee, not a hope

### 3. Code-Enforced Human Approval Before Writes

No card reaches Anki without an explicit terminal `APPROVE`:

- **Capability boundary**: The `Knowledge_Manager` only receives `fetch_siyuan_notes`; no agent receives an Anki write tool
- **Typed authorization**: The transcript is replayed through `PipelineRun`, and only exact protocol decisions advance state
- **Application gate**: `write_approved_run()` calls `PipelineRun.begin_write()` before invoking AnkiConnect
- **Invariant**: Reviewer approval, invalid human text, or malformed card JSON cannot authorize an Anki write

### 4. Application-Owned Side Effects

External writes are deliberately kept outside the LLM-controlled workflow:

- **Read path**: The `Knowledge_Manager` may call `fetch_siyuan_notes()` to read local source material
- **Write path**: Only application code may call `push_cards_batch()`, through `write_approved_run()`
- **Why**: Prompt instructions are not an authorization boundary; removing the tool from agents makes the rule enforceable
- **Current limitation**: AnkiConnect still returns legacy string errors; typed integration failures and idempotent recovery are planned hardening steps

### 5. Structured Observability via Logging

Each run writes a **`logs/{timestamp}.json`** file with full execution trace:

**Implementation** (`logger.py`):
```python
class PipelineLogger:
    def log_agent_message(agent: str, content: str, type: str) -> None
    def log_tool_call(agent: str, tool: str, input: dict, result: str) -> None
    def log_rejection(agent: str, reason: str) -> None  # Increments rejection_count, triggers guardrail at 2
    def save() -> Path  # Writes timestamped JSON file to logs/
```

**Example log entry:**
```json
{
  "timestamp": "2026-05-12T14:24:12.345678",
  "agent": "Card_Reviewer",
  "type": "rejection",
  "reason": "Back has multiple answers (violates MIP)",
  "rejection_count": 1,
  "guardrail_active": false
}
```

**Why this matters:**
- **Debugging**: Trace exactly what each agent said and why
- **Auditing**: Verify the decision path before cards were written to Anki
- **Guardrails**: See when rejection caps (2) triggered human escalation

## Features

- **Quality-First Card Generation**: Cards follow [SuperMemo's 20 Rules](https://supermemo.guru/wiki/20_rules_of_knowledge_formulation) (Minimum Information Principle)
- **Iterative Refinement**: Reviewer agent sends cards back for revision until they pass quality checks
- **Code-Enforced Human Oversight**: Agents cannot access the Anki write capability; exact human approval is required
- **Local-First**: Works with Ollama or any OpenAI-compatible LLM server - no cloud API keys required
- **GPU Acceleration**: Ollama 0.17+ supports Intel Arc, NVIDIA, and AMD GPUs
- **Structured Logging**: JSON traces of every run for debugging and auditing
- **Pydantic Validation**: Cards validated against schema before writing to Anki (prevents malformed data)

## Testing

The core reliability suite runs without Siyuan, Anki, an LLM server, or AutoGen:

```bash
python -m pytest
```

The tests cover:

- typed workflow transitions and exact approval parsing
- reviewer rejection and escalation behavior
- transcript replay from untrusted agent output
- invalid/malformed card output
- deterministic routing policy
- the agent capability policy (`Knowledge_Manager` has no Anki write tool)
- the hard invariant that an unapproved run performs **zero Anki writes**

For a fast syntax check:

```bash
python -m compileall -q main.py src tests
```

## Architecture Diagram

```mermaid
flowchart TD
    Siyuan[(Siyuan Notes)]
    Anki[(AnkiConnect)]

    subgraph Agents["Multi-Agent Conversation"]
        KM[Knowledge_Manager]
        CW[Card_Writer]
        CR[Card_Reviewer]
        Admin[/Admin Human Gate/]
    end

    subgraph Application["Deterministic Application Boundary"]
        Replay[Replay transcript into PipelineRun]
        Gate{can_write?}
        Writer[write_approved_run]
    end

    User((User)) --> KM
    KM -- "fetch_siyuan_notes()" --> Siyuan
    KM --> CW
    CW --> CR
    CR -- "REJECTED" --> CW
    CR -- "APPROVED" --> Admin
    Admin -- "REJECT" --> CW
    Admin -- "APPROVE" --> KM
    KM --> Done([TERMINATE])

    Agents --> Replay
    Replay --> Gate
    Gate -- "no" --> NoWrite([No side effect])
    Gate -- "yes" --> Writer
    Writer -- "push_cards_batch()" --> Anki
```

**Execution Flow:**
1. **Knowledge_Manager** reads source notes from Siyuan; it has no Anki write capability
2. **Card_Writer** creates flashcards following the Minimum Information Principle
3. **Card_Reviewer** validates quality → rejects poor cards (max 2 times, then escalates)
4. **Admin** provides an exact `APPROVE` or `REJECT` decision
5. The application replays the transcript into a typed **`PipelineRun`**
6. Only an authorized run reaches **`write_approved_run()`**, which owns the Anki side effect
7. **Logging** traces the conversation and application write result

## Quick Start

### Prerequisites

- **Python 3.13+**
- **[Ollama](https://ollama.com/)** - Local LLM inference (0.17+ recommended for Intel GPU support)
- **Siyuan Notes** with API enabled
- **Anki** with [AnkiConnect](https://ankiweb.net/shared/info/2055492159) plugin

### Installation

```bash
# Clone the repository
git clone https://github.com/ronketer/siyuan-to-anki.git
cd siyuan-to-anki

# Create virtual environment (using uv)
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv sync
```

### Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
notepad .env  # or your preferred editor
```

Required settings:
- TARGET_BLOCK_ID: The Siyuan block ID containing your notes

### Usage

1. Start Ollama and pull a model:
   ```bash
   ollama serve  # if not already running
   ollama pull qwen2.5-coder:3b  # Fast, reliable for multi-agent (~3B params)
   ```
2. Start Siyuan Notes
3. Start Anki (with AnkiConnect running)
4. Run the pipeline:

```bash
python main.py
```

> **Intel GPU Users**: Ollama 0.17+ automatically detects Intel Arc/Iris GPUs and uses Vulkan acceleration. No extra setup needed!

## Project Structure

```
.
+-- main.py                       # Entry point and application composition
+-- logs/                         # Execution traces (JSON per run)
+-- src/
|   +-- anki_pipeline/
|       +-- __init__.py
|       +-- agents.py             # AutoGen participant definitions only
|       +-- anki_writer.py        # Approval-gated application write boundary
|       +-- config.py             # Environment configuration
|       +-- logger.py             # Structured logging for observability
|       +-- models.py             # Pydantic flashcard models
|       +-- orchestrator.py       # Transcript -> trusted PipelineRun state
|       +-- routing.py            # Deterministic routing and tool policy
|       +-- tools.py              # Siyuan/Anki HTTP adapters
|       +-- workflow.py           # Typed workflow state machine
+-- tests/                        # Offline reliability/invariant tests
+-- pyproject.toml                # Dependencies and project metadata
+-- README.md
```

## Observability & Debugging

Every pipeline run produces a **`logs/{timestamp}.json`** file with a complete execution trace:

```json
{
  "run_id": "2026-05-12T14:23:45.123456",
  "entries": [
    {
      "timestamp": "2026-05-12T14:23:45.234567",
      "agent": "Card_Reviewer",
      "type": "rejection",
      "reason": "Back has multiple answers (violates MIP)",
      "rejection_count": 1
    },
    {
      "timestamp": "2026-05-12T14:24:12.345678",
      "agent": "Admin",
      "type": "approval",
      "card_count": 8
    }
  ]
}
```

**Why this matters:**
- **Debugging**: Trace exactly what each agent said and why
- **Auditing**: Verify the decision path before cards were written
- **Guardrails**: See when rejection caps triggered human escalation

## Flashcard Quality Standards

Cards are validated against [SuperMemo's 20 Rules of Formulating Knowledge](https://www.supermemo.com/en/blog/twenty-rules-of-formulating-knowledge):

1. **Minimum Information Principle**: One fact per card
2. **No Sets**: Avoid asking for lists of items
3. **Cloze Format**: Use fill-in-the-blank for complex facts
4. **Clean Text**: No formatting artifacts or tags

## LLM Configuration

### Model Requirements

This pipeline requires a model with **strong instruction-following** capabilities to properly execute multi-agent workflows. Smaller models (< 4B parameters) may skip agents or ignore the reflection loop.

### Ollama (Recommended)

Best for local inference with GPU acceleration:

```env
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL_ID=qwen2.5-coder:3b  # ~3B params, fast and reliable
```

> ⚠️ **Model Size Warning**: Models smaller than ~4B parameters may not follow multi-agent workflows correctly. They tend to skip agents, ignore the reflection loop, or call tools with placeholder values. Use 4B+ parameter models for reliable results.

### OpenAI (Cloud)

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_ID=gpt-4o-mini
LLM_API_KEY=sk-your-api-key
```

### Other OpenAI-Compatible Servers

Works with LM Studio, vLLM, or any server exposing `/v1/chat/completions`:

```env
LLM_BASE_URL=http://127.0.0.1:YOUR_PORT/v1
LLM_MODEL_ID=your-model-name
```

## Security Considerations

- **No credentials in code**: All secrets loaded from environment variables
- **Local inference**: Default configuration keeps all data on your machine
- **No sensitive data in prompts**: Knowledge content stays within local network
- **Token-based auth**: Siyuan API uses local token authentication

## Technologies

- [AutoGen 0.4+](https://microsoft.github.io/autogen/) - Multi-agent orchestration with `SelectorGroupChat`
- [Pydantic v2](https://docs.pydantic.dev/) - Schema validation before Anki writes
- [Ollama](https://ollama.com/) - Local LLM inference with GPU acceleration (Intel Arc, NVIDIA, AMD)
- [Siyuan Notes](https://b3log.org/siyuan/) - Local-first knowledge management via REST API
- [AnkiConnect](https://ankiweb.net/shared/info/2055492159) - Anki card creation via HTTP API

## License

MIT License - see [LICENSE](LICENSE) for details.

