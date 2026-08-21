"""Builds the system instruction that puts Gemini in charge of the
conversation.

Design decision (do not undo): the PROMPT drives the call, not a state
machine here. Gemini decides what to ask, when, and how to phrase it, from
this instruction and its own context. This module's only job is to hand it
an accurate picture of the question set and what's already answered.
"""

from __future__ import annotations

from glad.agent.script import QuestionSet
from glad.agent.state import SessionState

_INTRO = (
    "You are Glad, a helpful voice assistant sitting in on a live sales "
    "discovery call. Have a natural conversation, and over the course of "
    "it, get answers to the discovery questions listed below. Decide for "
    "yourself what to ask and when, and phrase questions naturally in "
    "context -- do not read the list verbatim and do not force a fixed "
    "order."
)

_ENGAGEMENT_RULES = (
    "You join the call DORMANT and stay silent until someone addresses "
    "you by name, or until you are told the floor is free and it is time "
    "to ask the next outstanding question. While an exchange with you is "
    "continuing -- a follow-up, a probe, a clarification -- call "
    "stay_engaged so you may keep speaking. Call go_dormant only on "
    "explicit dismissal ('thanks Glad', 'that's all') or when every "
    "question below has an answer. Recording answers is INDEPENDENT of "
    "engagement: always call record_answer when a participant answers a "
    "question, whether you are speaking or silent. When the script is "
    "complete or you are dismissed, you may say one short line (for "
    "example \"I'll go quiet, say Glad if you need me\") before going "
    "dormant — but only if people should hear it. go_dormant cuts off "
    "any audio that has not already been played: if the tool runs first, "
    "the rest of that turn is dropped and the room hears silence. Speak "
    "the line to completion, then call go_dormant. If you want to shut "
    "up immediately with no sign-off, call go_dormant first (or only). "
    "Do this ONLY for those two clean cases. Do NOT announce dormancy "
    "on topic drift."
)

_TOOL_RULES = (
    "Whenever a participant answers ANY of the questions below -- "
    "including one you have not asked yet, or one you already have an "
    "answer for -- call record_answer with its question_id and the "
    "answer's value, right then. This is what handles answers that come "
    "out of order; do not wait to ask about it yourself first. Recording "
    "an answer and acknowledging it out loud are independent: call "
    "record_answer even in turns where you don't say anything back about "
    "it, and even while dormant. If someone refines, corrects, or adds "
    "detail to a previous answer, call record_answer again with the same "
    "question_id -- it overwrites the old value, it does not create a "
    "duplicate."
)

_ALL_ANSWERED = "All questions are already answered. Do not re-ask any of them."

_BOT_NAMES = frozenset({"glad"})


def present_people(state: SessionState) -> list[str]:
    """Display names of humans currently in the call, host tagged."""
    present = [
        p for p in state.roster.present() if p.name.strip().lower() not in _BOT_NAMES
    ]
    present.sort(key=lambda p: p.name.lower())
    return [f"{p.name} (host)" if p.is_host else p.name for p in present]


def roster_context_line(state: SessionState) -> str | None:
    """One-shot context for Live: names without asking the model to speak."""
    names = present_people(state)
    if not names:
        return None
    return (
        "[Context only — people currently in this call: "
        + ", ".join(names)
        + ". Do not reply to this message. If asked who is here, name these people.]"
    )


def _roster_lines(state: SessionState) -> list[str]:
    names = present_people(state)
    lines = ["People in this call:"]
    if not names:
        lines.append("- (none identified yet)")
        return lines
    for name in names:
        lines.append(f"- {name}")
    lines.append(
        "You know these names. If asked who is here, list them. "
        "The audio you hear is mixed. Markers of the form "
        "'[Name is speaking]' identify the current voice — address that "
        "person and attribute answers to them. Do not reply to those markers."
    )
    return lines


def build_system_instruction(question_set: QuestionSet, state: SessionState) -> str:
    """Build the full system instruction for `question_set`/`state`. Called
    on every (re)connect so a resumed session, which loses its own
    context, avoids re-asking answered questions and sees the current roster.

    Gemini Live only accepts system_instruction at session setup. Roster
    changes mid-call land here on the next natural reconnect -- we do not
    reconnect just to refresh names.
    """
    lines = [_INTRO, ""]
    lines.extend(_roster_lines(state))
    lines.extend(["", "Questions:"])
    for question in question_set.questions:
        answer = state.answers.get(question.id)
        if answer is not None:
            lines.append(f"- [ANSWERED] {question.id}: {question.text} -> {answer.value!r}")
        else:
            lines.append(f"- [PENDING] {question.id}: {question.text}")

    remaining = state.remaining()
    lines.append("")
    if remaining:
        remaining_ids = ", ".join(q.id for q in remaining)
        lines.append(f"Still needed: {remaining_ids}.")
    else:
        lines.append(_ALL_ANSWERED)

    lines.append("")
    lines.append(_ENGAGEMENT_RULES)
    lines.append("")
    lines.append(_TOOL_RULES)
    return "\n".join(lines)
