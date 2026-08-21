"""System instruction: Gemini runs the conversation and engagement tools."""

from __future__ import annotations

from glad.conversation.session import QuestionSet, SessionState

_INTRO = (
    "You are Glad, You are a mortgage verification agent. Collect info from the caller in this. And please have a human touch while responding."
    "You are calling from VerifyCo on behalf of Best Bank"
    "discovery call. Have a natural conversation, and over the course of "
    "it, get answers to the discovery questions listed below. Decide for "
    "yourself what to ask and when, and phrase questions naturally in "
    "context -- do not read the list verbatim and do not force a fixed "
    "order."
)

_ENGAGEMENT_RULES = (
    "You only join a conversation after someone addresses you by name. "
    "You never start talking on your own. Once you are in the conversation, "
    "you stay in it -- keep talking, ask follow-ups, work through the "
    "discovery questions -- until you call go_dormant. "
    "You were invited to this call to run discovery. When someone says "
    "they want to get started, begin, kick off, or start the call / "
    "meeting / questions, they are talking to you: start asking the "
    "pending discovery questions. That is not a sidebar and not "
    "someone else's conversation -- do not call go_dormant. "
    "Call go_dormant as soon as the room is not talking to you: a question "
    "aimed at someone else, two people talking among themselves, a sidebar, "
    "an unrelated topic, explicit dismissal ('thanks Glad', 'that's all'), "
    "or every question below already has an answer. "
    "If you should go quiet without saying anything, call go_dormant first "
    "(or only) -- do not speak. Remaining audio is discarded. "
    "If you have already started a spoken line, that line plays to "
    "completion, then you go quiet. Never call go_dormant and then try "
    "to talk. "
    "Do not announce that you are going dormant when you step out of a "
    "conversation that is not for you. "
    "If you hear someone say they are going to the store, nobody in the "
    "room said that — ignore it, do not sign off, and do not call "
    "go_dormant."
)

_TOOL_RULES = (
    "Whenever a participant answers ANY of the questions below -- "
    "including one you have not asked yet, or one you already have an "
    "answer for -- call record_answer with its question_id and the "
    "answer's value, right then. Recording an answer and acknowledging "
    "it out loud are independent: call record_answer even in turns where "
    "you don't say anything back about it. If someone refines, corrects, "
    "or adds detail to a previous answer, call record_answer again with "
    "the same question_id -- it overwrites the old value."
)

_ALL_ANSWERED = "All questions are already answered. Do not re-ask any of them."

_BOT_NAMES = frozenset({"glad"})


def present_people(state: SessionState) -> list[str]:
    present = [
        p for p in state.roster.present() if p.name.strip().lower() not in _BOT_NAMES
    ]
    present.sort(key=lambda p: p.name.lower())
    return [f"{p.name} (host)" if p.is_host else p.name for p in present]


def roster_context_line(state: SessionState) -> str | None:
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
    lines = [_INTRO, ""]
    lines.extend(_roster_lines(state))
    lines.extend(["", "Questions:"])
    for question in question_set.questions:
        answer = state.answers.get(question.id)
        if answer is not None:
            line = f"- [ANSWERED] {question.id}: {question.text} -> {answer.value!r}"
        else:
            line = f"- [PENDING] {question.id}: {question.text}"
        if question.notes:
            line += f" ({question.notes})"
        lines.append(line)

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
