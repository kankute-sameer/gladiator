"""Roster: join/leave retention, prompt listing, and Recall envelope decode."""

from __future__ import annotations

from glad.conversation.prompt import build_system_instruction, roster_context_line
from glad.conversation.session import Question, QuestionSet, Roster, SessionState
from glad.transport import participants as participants_ws

_QUESTION_SET = QuestionSet(
    id="test_set",
    version=1,
    questions=(Question(id="budget", text="What's your budget?"),),
)


def _envelope(event: str, participant_id: int, name: str, is_host: bool | None = None) -> dict:
    return {
        "event": event,
        "data": {
            "data": {
                "participant": {
                    "id": participant_id,
                    "name": name,
                    "is_host": is_host,
                    "platform": None,
                    "extra_data": None,
                    "email": None,
                },
                "timestamp": {"absolute": "2026-08-21T10:00:00Z", "relative": 1.0},
                "data": None,
            }
        },
    }


def test_join_then_leave_keeps_the_person_with_left_at() -> None:
    clock = iter([10.0, 20.0]).__next__
    roster = Roster(_clock=clock)
    roster.join(1, "Alice", True)
    roster.leave(1, "Alice", True)

    person = roster.get(1)
    assert person is not None
    assert person.name == "Alice"
    assert person.is_host is True
    assert person.joined_at == 10.0
    assert person.left_at == 20.0
    assert roster.present() == []
    assert len(roster.all()) == 1


def test_rejoin_clears_left_at_without_dropping_the_record() -> None:
    clock = iter([1.0, 2.0, 3.0]).__next__
    roster = Roster(_clock=clock)
    roster.join(1, "Alice", False)
    roster.leave(1, "Alice", False)
    roster.join(1, "Alice", False)

    person = roster.get(1)
    assert person is not None
    assert person.left_at is None
    assert person.joined_at == 1.0
    assert [p.id for p in roster.present()] == [1]


def test_note_is_true_only_for_first_sighting_and_rejoin() -> None:
    roster = Roster(_clock=lambda: 1.0)
    assert roster.note(1, "Alice") is True
    assert roster.note(1, "Alice") is False
    roster.leave(1, "Alice", None)
    assert roster.note(1, "Alice") is True


def test_unknown_leave_is_still_retained() -> None:
    roster = Roster(_clock=lambda: 5.0)
    person = roster.leave(9, "Bob", None)
    assert person.left_at == 5.0
    assert roster.get(9) is person


def test_prompt_lists_current_people_and_marks_host() -> None:
    state = SessionState(session_id="s1", question_set=_QUESTION_SET)
    state.roster.join(1, "Alice", True, now=1.0)
    state.roster.join(2, "Bob", False, now=2.0)
    state.roster.join(3, "Glad", False, now=3.0)
    state.roster.leave(2, "Bob", False, now=4.0)

    instruction = build_system_instruction(_QUESTION_SET, state)
    assert "- Alice (host)" in instruction
    assert "Bob" not in instruction
    assert "- Glad" not in instruction
    assert "People in this call:" in instruction


def test_prompt_empty_roster_says_none_identified() -> None:
    state = SessionState(session_id="s1", question_set=_QUESTION_SET)
    instruction = build_system_instruction(_QUESTION_SET, state)
    assert "(none identified yet)" in instruction
    assert "go_dormant" in instruction
    assert "stay_engaged" not in instruction
    assert "discarded" in instruction
    assert "get started" in instruction


def test_roster_context_line_lists_people_and_skips_glad() -> None:
    state = SessionState(session_id="s1", question_set=_QUESTION_SET)
    assert roster_context_line(state) is None
    state.roster.join(1, "Alice", True, now=1.0)
    state.roster.join(2, "Glad", False, now=2.0)
    line = roster_context_line(state)
    assert line is not None
    assert "Alice (host)" in line
    assert "Glad" not in line
    assert "Do not reply" in line


def test_parse_join_leave_update_and_ignores_speech() -> None:
    join = participants_ws._parse_event(_envelope("participant_events.join", 1, "Alice", True))
    assert join is not None
    assert join.kind == "join"
    assert join.participant_id == 1
    assert join.name == "Alice"
    assert join.is_host is True

    leave = participants_ws._parse_event(_envelope("participant_events.leave", 1, "Alice", True))
    assert leave is not None and leave.kind == "leave"

    update = participants_ws._parse_event(_envelope("participant_events.update", 1, "Alice K", True))
    assert update is not None and update.kind == "update" and update.name == "Alice K"

    assert participants_ws._parse_event(_envelope("participant_events.speech_on", 1, "Alice")) is None
    assert participants_ws._parse_event(_envelope("transcript.data", 1, "Alice")) is None


def test_transport_apply_updates_roster_and_is_idempotent_on_duplicate_join() -> None:
    roster = Roster(_clock=lambda: 42.0)
    participants_ws.set_roster(roster)
    try:
        participants_ws._apply(participants_ws._parse_event(_envelope("participant_events.join", 1, "Alice", True)))
        participants_ws._apply(participants_ws._parse_event(_envelope("participant_events.join", 1, "Alice", True)))
        assert len(roster) == 1
        assert roster.get(1).joined_at == 42.0

        participants_ws._apply(participants_ws._parse_event(_envelope("participant_events.leave", 1, "Alice", True)))
        assert roster.get(1).left_at == 42.0
    finally:
        participants_ws.set_roster(None)


def test_update_of_unknown_person_counts_as_a_join() -> None:
    roster = Roster(_clock=lambda: 7.0)
    participants_ws.set_roster(roster)
    try:
        participants_ws._apply(
            participants_ws._parse_event(_envelope("participant_events.update", 4, "Cara", False))
        )
        person = roster.get(4)
        assert person is not None
        assert person.name == "Cara"
        assert person.left_at is None
    finally:
        participants_ws.set_roster(None)
