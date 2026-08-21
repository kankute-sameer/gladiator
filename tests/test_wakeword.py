"""Tests for glad.agent.wakeword: stage-1 recall and stage-2 vocative
disambiguation."""

from __future__ import annotations

import pytest

from glad.agent.wakeword import Stage2Verdict, describe_verdict, detect

# Real vocatives: Glad is being addressed and should wake.
_REAL_VOCATIVES = [
    "Glad, what do you think about that?",
    "Glad, can you check the budget question?",
    "Hey Glad, are you still there?",
    "Glad, what's your take on this?",
    "Glad, tell us what you heard.",
    "Glad, how does that sound?",
    "glad you there",
    "Glad, you there?",
    "Hey Glad, you there?",
    "gladiator",
    "Glad iator, you there?",
]

# Adjectival / politeness uses: must be suppressed, never wake.
_ADJECTIVAL_USES = [
    "glad we could connect",
    "I'd be glad to send that over",
    "so glad you asked",
    "really glad you called",
    "I'm glad that worked out",
    "we're glad you joined",
    "glad it all worked out",
    "glad about the update",
    "very glad we chose this vendor",
    "glad to help with that",
    "glad you called",
    "I'm glad you're here",
    "i'm glad you're here",
    "I am so glad about this deal",
    "pretty glad we found this tool",
    "quite glad it's finally sorted",
    "we were glad that you called back",
]


@pytest.mark.parametrize("utterance", _REAL_VOCATIVES)
def test_real_vocative_wakes(utterance: str) -> None:
    result = detect(utterance)
    assert result.woken, f"expected wake for {utterance!r}, got verdict={result.verdict}"


@pytest.mark.parametrize("utterance", _ADJECTIVAL_USES)
def test_adjectival_use_suppressed(utterance: str) -> None:
    result = detect(utterance)
    assert not result.woken, f"expected suppression for {utterance!r}, got verdict={result.verdict}"
    assert result.verdict in (
        Stage2Verdict.SUPPRESSED_NEGATIVE_FOLLOWER,
        Stage2Verdict.SUPPRESSED_NEGATIVE_PRECEDER,
        Stage2Verdict.SUPPRESSED_NO_POSITIVE_SIGNAL,
    )


def test_fixture_covers_at_least_twenty_utterances() -> None:
    assert len(_REAL_VOCATIVES) + len(_ADJECTIVAL_USES) >= 20


def test_gladiator_wakes_and_does_not_also_match_glad() -> None:
    result = detect("gladiator")
    assert result.woken
    assert result.stage1_matches[0].phrase == "gladiator"
    assert len(result.stage1_matches) == 1


def test_glad_iator_split_across_tokens_wakes() -> None:
    result = detect("glad iator")
    assert result.woken
    assert result.stage1_matches[0].phrase == "glad iator"


def test_no_match_returns_none_verdict() -> None:
    result = detect("let's talk about the budget for next quarter")
    assert result.stage1_matches == ()
    assert result.verdict is None
    assert not result.woken


def test_asr_neighbours_clad_and_glide_do_not_wake() -> None:
    assert not detect("the walls are clad in oak panels").woken
    assert not detect("let the presentation glide through the slides").woken


def test_case_and_punctuation_normalized() -> None:
    assert detect("GLAD! What do you think?").woken
    assert not detect("So GLAD, you could make it.").woken


def test_utterance_initial_position_is_a_positive_signal_alone() -> None:
    # No interrogative/imperative follows, but utterance-initial position
    # alone is enough per the stage-2 rule list. (Deliberately avoids
    # "glad we"/"glad you"/etc -- those are negative-follower patterns
    # that suppress regardless of position, per the explicit rule list;
    # see test_adjectival_use_suppressed with "glad we could connect".)
    result = detect("Glad, quick question for you.")
    assert result.woken


def test_mid_utterance_without_positive_signal_is_suppressed() -> None:
    result = detect("anyway glad works fine for now I guess")
    assert not result.woken
    assert result.verdict is Stage2Verdict.SUPPRESSED_NO_POSITIVE_SIGNAL


def test_suppression_reason_is_none_when_accepted_or_no_match() -> None:
    assert detect("Glad, are you there?").suppression_reason is None
    assert detect("let's talk about the budget").suppression_reason is None


def test_suppression_reason_is_the_verdict_value_when_suppressed() -> None:
    result = detect("so glad you called")
    assert result.suppression_reason == result.verdict.value == "suppressed_negative_preceder"


def test_describe_verdict_has_a_human_reason_for_every_suppression() -> None:
    for verdict in Stage2Verdict:
        if verdict is Stage2Verdict.ACCEPTED:
            continue
        assert describe_verdict(verdict)  # non-empty for every suppression case
    assert describe_verdict(None) == "no match"
