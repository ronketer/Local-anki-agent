#!/usr/bin/env python3
'''Autonomous Local Knowledge to Anki Pipeline.

A multi-agent AI system that extracts knowledge from Siyuan Notes
and creates optimized Anki flashcards using AutoGen.

This project demonstrates:
- Multi-agent orchestration with specialized roles
- Tool use for external API integration
- Reflection pattern for quality assurance
- Human-in-the-loop approval workflow
- Local-first, privacy-preserving AI (no cloud LLM dependencies)

Usage:
    python main.py                  # Run pipeline with human review
    python main.py --block ID       # Override target block ID
'''

import argparse
import asyncio
import json
import re
import sys

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console

from src.anki_pipeline.agents import create_agents, create_model_client
from src.anki_pipeline.anki_writer import write_approved_run
from src.anki_pipeline.config import config
from src.anki_pipeline.errors import ConfigurationError, PipelineError
from src.anki_pipeline.logger import PipelineLogger
from src.anki_pipeline.orchestrator import replay_workflow
from src.anki_pipeline.retry import retry_call
from src.anki_pipeline.routing import selector_func
from src.anki_pipeline.workflow import HumanDecision, parse_human_decision


def parse_args() -> argparse.Namespace:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Generate Anki flashcards from Siyuan Notes using AI agents.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--block', '-b', type=str,
        help='Override TARGET_BLOCK_ID from .env',
    )
    return parser.parse_args()


# ANSI color codes
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
MAGENTA = '\033[35m'
BLUE = '\033[34m'
RESET = '\033[0m'
DIM = '\033[2m'

AGENT_STYLES = {
    'user': ('[USER]', CYAN),
    'Knowledge_Manager': ('[KNOWLEDGE MANAGER]', YELLOW),
    'Card_Writer': ('[CARD WRITER]', GREEN),
    'Card_Reviewer': ('[CARD REVIEWER]', MAGENTA),
    'Admin': ('[ADMIN]', BLUE),
}


def extract_json_cards(content: str) -> dict | None:
    '''Extract JSON cards from content that may have surrounding text.'''
    # Try to find JSON in markdown code blocks or raw
    patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',  # Markdown code block
        r'(\{"cards":\s*\[.*?\]\})',         # Raw JSON
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return None


def format_markdown(text: str) -> str:
    '''Format markdown text for terminal display.'''
    # Add blank lines around headers
    text = re.sub(r'(^|\n)(#{1,3})\s+(.+)', r'\1\n\2 \3\n', text)
    # Bold headers
    text = re.sub(r'^(#{1,3})\s+(.+)$', rf'{YELLOW}\2{RESET}', text, flags=re.MULTILINE)
    # Format code blocks
    text = re.sub(r'```(\w+)?\n', f'{DIM}', text)
    text = re.sub(r'```', f'{RESET}', text)
    return text


def format_cards_display(cards: list) -> str:
    '''Format cards for clear display.'''
    lines = []
    for i, card in enumerate(cards, 1):
        front = card.get('front', card.get('question', ''))
        back = card.get('back', card.get('answer', ''))
        lines.append(f'  {DIM}Card {i}:{RESET}')
        lines.append(f'    Q: {front}')
        lines.append(f'    A: {GREEN}{back}{RESET}')
    return '\n'.join(lines)


def format_agent_message(source: str, content: str) -> str:
    '''Format a message with agent name header.'''
    name, color = AGENT_STYLES.get(source, (source, RESET))
    header = f'\n{color}{"-"*50}\n{name}\n{"-"*50}{RESET}'

    # Pretty-print cards if Card_Writer
    if source == 'Card_Writer':
        data = extract_json_cards(content)
        if data and 'cards' in data:
            return f'{header}\n{format_cards_display(data["cards"])}'

    return f'{header}\n{content}'


async def main() -> int:
    '''Run the flashcard generation pipeline.'''
    # Initialize logger for observability
    logger = PipelineLogger()
    
    # Parse CLI arguments
    args = parse_args()

    # Override block ID if provided
    if args.block:
        config.TARGET_BLOCK_ID = args.block

    # Validate configuration
    try:
        config.require_valid()
    except ConfigurationError as exc:
        print(f'Configuration Error: {exc}')
        print('\nPlease check your .env file or provide --block.')
        logger.log_agent_message(
            'Application',
            json.dumps(exc.as_dict(), ensure_ascii=False),
            'configuration_error',
        )
        logger.log_outcome('error', saved_cards=0)
        logger.save()
        return 1

    print(f'''
{CYAN}===================================================
   Autonomous Knowledge to Anki Pipeline
==================================================={RESET}
  Target: {config.TARGET_BLOCK_ID}
  Model:  {config.LLM_MODEL_ID}

{DIM}Workflow: Fetch -> Write -> Review -> [You Approve] -> Save to Anki
When prompted, type exactly: APPROVE or REJECT{RESET}
''')

    model_client = create_model_client()
    agents = create_agents(model_client)

    # Pre-fetch content to work around models that struggle with tool calling
    from src.anki_pipeline.tools import fetch_siyuan_notes
    print(f'{DIM}Fetching content from Siyuan...{RESET}')
    logger.log_agent_message(
        'Knowledge_Manager',
        f'Fetching block {config.TARGET_BLOCK_ID}',
        'tool_call',
    )
    try:
        content = retry_call(
            lambda: fetch_siyuan_notes(config.TARGET_BLOCK_ID)
        )
    except PipelineError as exc:
        print(
            f'{YELLOW}Warning: Siyuan prefetch failed '
            f'[{exc.code}, retryable={exc.retryable}]: {exc}{RESET}'
        )
        print('Continuing anyway - the agent workflow may attempt to fetch...')
        logger.log_agent_message(
            'Application',
            json.dumps(exc.as_dict(), ensure_ascii=False),
            'integration_error',
        )
        prefetched_content = None
    else:
        # Parse and extract just the markdown
        try:
            data = json.loads(content)
            prefetched_content = data.get('kramdown', content)
            print(f'{GREEN}Content fetched successfully ({len(prefetched_content)} chars){RESET}\n')
            logger.log_tool_call(
                'Knowledge_Manager',
                'fetch_siyuan_notes',
                {'block_id': config.TARGET_BLOCK_ID},
                f'Success ({len(prefetched_content)} chars)'
            )
        except json.JSONDecodeError:
            prefetched_content = content

    team = SelectorGroupChat(
        participants=[
            agents['knowledge_manager'],
            agents['card_writer'],
            agents['card_reviewer'],
            agents['admin'],
        ],
        model_client=model_client,
        selector_func=selector_func,
        termination_condition=(
            TextMentionTermination('TERMINATE') |
            MaxMessageTermination(30)  # Safety limit
        ),
    )

    # Build task with prefetched content if available
    if prefetched_content:
        task = (
            f"Here is the content from Siyuan Notes:\n\n"
            f"---\n{prefetched_content}\n---\n\n"
            f"Create Anki flashcards from this content. "
            f"Card_Writer: draft the cards. Card_Reviewer: review them. "
            f"Admin will approve. The application saves cards only after the agent workflow ends."
        )
    else:
        task = (
            f"Fetch the notes for Siyuan block ID '{config.TARGET_BLOCK_ID}'. "
            'Draft and review the Anki cards. Admin will provide the final approval.'
        )

    # Run pipeline and capture result
    result = await Console(team.run_stream(task=task))

    # Log all messages from the pipeline
    for msg in result.messages:
        if hasattr(msg, 'source') and hasattr(msg, 'content'):
            source = msg.source
            content = str(msg.content)[:500]  # Truncate for logging
            
            if source == 'Card_Reviewer':
                if 'REJECTED' in content:
                    logger.log_rejection(source, content)
                elif 'APPROVED' in content:
                    logger.log_agent_message(source, content, 'approval_check')
            elif source == 'Admin':
                decision = parse_human_decision(content)
                if decision == HumanDecision.APPROVED:
                    logger.log_approval(0)
                elif decision == HumanDecision.REJECTED:
                    logger.log_rejection(source, content)
                else:
                    logger.log_agent_message(source, content, 'invalid_human_decision')
            else:
                logger.log_agent_message(source, content, 'message')

    # Rebuild authoritative application state from the untrusted conversation.
    # Only exact protocol decisions and validated card JSON advance the state.
    run = replay_workflow(config.TARGET_BLOCK_ID, result.messages)

    saved_card_count = 0
    write_failed = False

    if run.can_write:
        print(f'\n{YELLOW}Saving explicitly approved cards...{RESET}')
        try:
            save_result = write_approved_run(run)
            saved_card_count = len(run.cards.cards)
            print(f'{GREEN}{save_result}{RESET}')
            logger.log_tool_call(
                'Application',
                'write_approved_run',
                {'run_id': run.run_id, 'card_count': saved_card_count},
                save_result,
            )
        except PipelineError as exc:
            write_failed = True
            print(
                f'{YELLOW}Anki write failed '
                f'[{exc.code}, retryable={exc.retryable}]: {exc}{RESET}'
            )
            logger.log_agent_message(
                'Application',
                json.dumps(exc.as_dict(), ensure_ascii=False),
                'integration_error',
            )
    else:
        print(
            f'{DIM}No Anki write performed '
            f'(stage={run.stage.value}, approved={run.human_decision.value}).{RESET}'
        )

    print(f'\n{CYAN}Pipeline complete.{RESET}')
    logger.log_outcome('error' if write_failed else 'success', saved_cards=saved_card_count)
    log_path = logger.save()
    print(f'{DIM}Log saved: {log_path}{RESET}')
    
    return 1 if write_failed else 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
