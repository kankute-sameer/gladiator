"""Per-participant audio bookkeeping: who is currently talking, and what
their frames sum to once mixed and noise-gated."""

from __future__ import annotations

from dataclasses import dataclass, field

from glad.audio.pcm import mix, rms


@dataclass(slots=True)
class ParticipantActivity:
    """Tracks each participant's latest frame, arrival time, and RMS,
    evicting anyone who's gone quiet for `stale_after_s`."""

    stale_after_s: float
    _pcm: dict[int, bytes] = field(default_factory=dict)
    _seen_at: dict[int, float] = field(default_factory=dict)
    _rms: dict[int, float] = field(default_factory=dict)

    def observe(self, participant_id: int, pcm: bytes, now: float) -> float:
        """Record one frame and evict stale participants. Returns this
        frame's RMS level."""
        level = rms(pcm)
        self._rms[participant_id] = level
        self._pcm[participant_id] = pcm
        self._seen_at[participant_id] = now
        self._evict_stale(now)
        return level

    def _evict_stale(self, now: float) -> None:
        for participant_id, seen_at in list(self._seen_at.items()):
            if now - seen_at > self.stale_after_s:
                del self._seen_at[participant_id]
                del self._pcm[participant_id]
                self._rms.pop(participant_id, None)

    def mix(self, gate_threshold: float) -> bytes:
        """Sum every live participant's latest frame into one mixed frame."""
        return mix(list(self._pcm.values()), gate_threshold=gate_threshold)

    def anyone_active(self, threshold: float, now: float) -> bool:
        """True if any non-stale participant's last frame was at/above `threshold`."""
        return any(level >= threshold for level in self.levels(now).values())

    def levels(self, now: float) -> dict[int, float]:
        """Stored RMS for participants still inside the stale window."""
        return {
            participant_id: self._rms[participant_id]
            for participant_id, seen_at in self._seen_at.items()
            if now - seen_at <= self.stale_after_s
        }
