"""Application configuration loaded from the environment / .env file."""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Glad meeting bot server."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    recall_api_key: SecretStr
    recall_base_url: str = "https://us-east-1.recall.ai"
    public_url: str
    port: int = 8000
    log_level: str = "INFO"
    # Persist mixed inbound PCM (what Gemini hears) as runs/<run>-inbound.wav
    record_inbound_audio: bool = True

    # Name of a file under question_sets/ (without .yaml).
    question_set: str = "discovery_v1"

    gemini_api_key: SecretStr
    gemini_model: str = "gemini-3.1-flash-live-preview"
    gemini_voice: str = "Kore"

    # TTL: how long Glad may keep speaking after it *finishes* a turn (the
    # clock does not run while Glad is talking). Hard cap bounds a span
    # even if stay_engaged keeps firing.
    engagement_ttl_s: float = 10.0
    engagement_hard_cap_s: float = 120.0
    # Self-initiate: ask the next script question after this much floor-free
    # silence, but not again until the cooldown elapses.
    self_initiate_gap_s: float = 3.0
    self_initiate_cooldown_s: float = 20.0
    # Drop frames below this RMS (0-1 full scale) before mixing.
    audio_gate_threshold: float = 0.02

    @property
    def public_ws_url(self) -> str:
        """`public_url` with the `wss://` scheme, for Recall's realtime
        endpoints connecting back over the same tunnel."""
        return self.public_url.replace("https://", "wss://", 1)


settings = Settings()
