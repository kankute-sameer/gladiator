"""Pure audio-domain math. No I/O, no state."""

from glad.audio.pcm import mix, rms

__all__ = ["mix", "rms"]
