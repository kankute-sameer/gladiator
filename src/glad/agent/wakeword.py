"""Two-stage wake word detection over FINAL transcript segments only.

Stage 1 (`_match_stage1`) is high recall: matches "glad" and its ASR
near-misses on word boundaries, longest phrase first so "gladiator" never
decomposes into a "glad" match.

Stage 2 (`_disambiguate`) answers the real question: was Glad addressed,
or is this the politeness word ("glad to help", "so glad you called")?
It's an explicit rule list rather than a single regex, so the rules
double as documentation of what counts as a vocative use.

Detection only answers "was I addressed" -- whether Glad may actually
speak right now is `glad.agent.floor`'s job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Longest-first: "gladiator" must be tried before "glad" or the shorter
# phrase would match its prefix. "gladiator"/"glad iator" ARE accepted as
# a vocative (common ASR of the name); "clad"/"glide" are not.
_STAGE1_PHRASES: tuple[str, ...] = tuple(
    sorted(
        [
            "gladiator",
            "glad iator",
            "glad",
            "clad",
            "glide",
        ],
        key=len,
        reverse=True,
    )
)

_WAKE_NEIGHBOURS = frozenset({"gladiator", "glad iator"})
_NON_ADDRESSING_NEIGHBOURS = frozenset({"clad", "glide"})

_WORD_RE = re.compile(r"[a-z0-9']+")

_INTERROGATIVES = frozenset(
    {"what", "who", "when", "where", "why", "how", "which", "is", "are", "can", "could", "would", "do", "does", "did"}
)
# Common sentence-initial imperative verbs directed at an assistant.
_IMPERATIVES = frozenset(
    {
        "tell",
        "give",
        "help",
        "ask",
        "explain",
        "walk",
        "show",
        "remind",
        "check",
        "record",
        "note",
        "repeat",
        "summarize",
        "confirm",
    }
)

# Explicit, readable suppression rules: (phrase-to-look-for-immediately-after-match).
# Each is checked as "glad" (or its matched span) directly followed by one of
# these tokens -- the textbook adjectival/politeness uses of the word.
_NEGATIVE_FOLLOWERS: tuple[str, ...] = (
    "to",
    "that",
    "we",
    "you",
    "you're",
    "youre",
    "it",
    "about",
)
# Intensifiers immediately BEFORE "glad" that mark it as the adjective
# ("so glad", "really glad") rather than a name being addressed.
_NEGATIVE_PRECEDERS: tuple[str, ...] = (
    "so",
    "really",
    "very",
    "pretty",
    "quite",
    "i'm",
    "im",
    "am",
)

# "Glad, you there?" is vocative. "glad you called" is not.
_PRESENCE_CUES = frozenset({"there", "here", "around", "listening", "ready", "awake", "up"})

_LOOKAHEAD_TOKENS = 4


class Stage2Verdict(str, Enum):
    ACCEPTED = "accepted"
    SUPPRESSED_NEGATIVE_FOLLOWER = "suppressed_negative_follower"
    SUPPRESSED_NEGATIVE_PRECEDER = "suppressed_negative_preceder"
    SUPPRESSED_NO_POSITIVE_SIGNAL = "suppressed_no_positive_signal"
    SUPPRESSED_NON_ADDRESSING_NEIGHBOUR = "suppressed_non_addressing_neighbour"


_VERDICT_EXPLANATIONS: dict[Stage2Verdict, str] = {
    Stage2Verdict.SUPPRESSED_NEGATIVE_FOLLOWER: "sounds like the word 'glad', not the name",
    Stage2Verdict.SUPPRESSED_NEGATIVE_PRECEDER: "sounds like 'so glad' / 'really glad', not the name",
    Stage2Verdict.SUPPRESSED_NO_POSITIVE_SIGNAL: "heard something like Glad, but not as an address",
    Stage2Verdict.SUPPRESSED_NON_ADDRESSING_NEIGHBOUR: "heard a similar word, not Glad's name",
}


def describe_verdict(verdict: Stage2Verdict | None) -> str:
    """Human-readable reason for a verdict, for logging."""
    if verdict is None:
        return "no match"
    return _VERDICT_EXPLANATIONS.get(verdict, verdict.value.replace("_", " "))


@dataclass(frozen=True, slots=True)
class Stage1Match:
    """One high-recall hit: which phrase matched, and where (token indices
    into the normalized, whitespace-split utterance)."""

    phrase: str
    start_token: int
    end_token: int  # exclusive


@dataclass(frozen=True, slots=True)
class WakeWordResult:
    text: str
    stage1_matches: tuple[Stage1Match, ...]
    verdict: Stage2Verdict | None  # None if stage 1 found nothing at all
    matched_phrase: str | None

    @property
    def woken(self) -> bool:
        return self.verdict is Stage2Verdict.ACCEPTED

    @property
    def suppression_reason(self) -> str | None:
        """`verdict.value` if stage 1 matched but the match was not
        accepted, else None."""
        if self.verdict is None or self.verdict is Stage2Verdict.ACCEPTED:
            return None
        return self.verdict.value


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse whitespace. Matching
    happens on this normalized form so "Glad," / "GLAD!" / "glad." all
    behave the same as bare "glad"."""
    lowered = text.lower()
    stripped = re.sub(r"[^a-z0-9'\s]", " ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def _tokenize(normalized: str) -> list[str]:
    return normalized.split(" ") if normalized else []


def _match_stage1(normalized: str) -> list[Stage1Match]:
    """Find every occurrence of any trigger phrase, longest first, on word
    boundaries. A span already claimed by a longer phrase is not
    re-matched by a shorter one within it."""
    tokens = _tokenize(normalized)
    claimed = [False] * len(tokens)
    matches: list[Stage1Match] = []

    for phrase in _STAGE1_PHRASES:  # already longest-first
        phrase_tokens = phrase.split(" ")
        n = len(phrase_tokens)
        for start in range(0, len(tokens) - n + 1):
            end = start + n
            if any(claimed[start:end]):
                continue
            if tokens[start:end] == phrase_tokens:
                matches.append(Stage1Match(phrase=phrase, start_token=start, end_token=end))
                for i in range(start, end):
                    claimed[i] = True

    matches.sort(key=lambda m: m.start_token)
    return matches


def _disambiguate(tokens: list[str], match: Stage1Match) -> Stage2Verdict:
    """Explicit rule list, checked in order. Each rule is one readable
    reason a match is suppressed or accepted -- this list is the spec."""
    if match.phrase in _NON_ADDRESSING_NEIGHBOURS:
        return Stage2Verdict.SUPPRESSED_NON_ADDRESSING_NEIGHBOUR
    if match.phrase in _WAKE_NEIGHBOURS:
        return Stage2Verdict.ACCEPTED

    preceding = tokens[match.start_token - 1] if match.start_token > 0 else None
    if preceding in _NEGATIVE_PRECEDERS:
        return Stage2Verdict.SUPPRESSED_NEGATIVE_PRECEDER

    following = tokens[match.end_token] if match.end_token < len(tokens) else None
    rest = tokens[match.end_token + 1 : match.end_token + _LOOKAHEAD_TOKENS]
    # "Glad, you there?" is vocative. "glad you're here" is the adjective.
    vocative_you = following == "you" and any(tok in _PRESENCE_CUES for tok in rest)
    if vocative_you:
        return Stage2Verdict.ACCEPTED
    if following in _NEGATIVE_FOLLOWERS:
        return Stage2Verdict.SUPPRESSED_NEGATIVE_FOLLOWER

    # Positive signal 1: utterance-initial position ("Glad, ..." / "Glad?").
    if match.start_token == 0:
        return Stage2Verdict.ACCEPTED

    # Positive signal 2: an interrogative or imperative in the next few
    # tokens. Presence words ("here"/"there") are not a signal on their
    # own — they only count in the vocative_you pattern above.
    lookahead = tokens[match.end_token : match.end_token + _LOOKAHEAD_TOKENS]
    if any(tok in _INTERROGATIVES or tok in _IMPERATIVES for tok in lookahead):
        return Stage2Verdict.ACCEPTED

    return Stage2Verdict.SUPPRESSED_NO_POSITIVE_SIGNAL


def detect(text: str) -> WakeWordResult:
    """Run both stages over one FINAL transcript segment. Only the first
    stage-1 match is disambiguated -- one verdict per utterance."""
    normalized = _normalize(text)
    tokens = _tokenize(normalized)
    stage1 = _match_stage1(normalized)
    if not stage1:
        return WakeWordResult(text=text, stage1_matches=(), verdict=None, matched_phrase=None)

    first = stage1[0]
    verdict = _disambiguate(tokens, first)
    return WakeWordResult(
        text=text,
        stage1_matches=tuple(stage1),
        verdict=verdict,
        matched_phrase=first.phrase,
    )
