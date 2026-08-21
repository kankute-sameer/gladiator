"""CLI: send a Glad bot into a meeting.

Usage:
    python -m scripts.send_bot <meet-url>
"""

from __future__ import annotations

import argparse
import asyncio

from glad.conversation.session import load_question_set
from glad.config import settings
from glad.logging import configure_logging, get_logger
from glad.recall.client import RecallClient

configure_logging(settings.log_level)
logger = get_logger(__name__)


async def send_bot(meeting_url: str) -> str:
    client = RecallClient()
    return await client.create_bot(meeting_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a Glad bot into a meeting.")
    parser.add_argument("meeting_url", help="The Google Meet URL to join.")
    parser.add_argument(
        "--question-set",
        default=settings.question_set,
        help=(
            "Name of a file under question_sets/ (without .yaml) driving the "
            "discovery conversation. Must match the server's QUESTION_SET -- "
            "this only validates the name locally and fails fast before "
            f"creating a bot (default: {settings.question_set!r})."
        ),
    )
    args = parser.parse_args()

    # Validate before any bot is created: a bad question-set name should
    # never result in a bot silently joining a call the server can't run.
    question_set = load_question_set(args.question_set)
    logger.info("Using question set %s (v%d)", question_set.id, question_set.version)

    bot_id = asyncio.run(send_bot(args.meeting_url))
    logger.info("Bot id: %s", bot_id)
    logger.info("Public URL: %s", settings.public_url)


if __name__ == "__main__":
    main()
