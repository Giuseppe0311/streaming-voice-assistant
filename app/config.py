"""Validated settings loaded only when the application starts."""
import os
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    voice_enabled: bool = False
    transcription_model: str = 'gpt-live-transcribe'
    llm_model: str = 'gpt-5.6-luna'
    tts_model: str = 'gpt-4o-mini-tts'
    tts_voice: str = 'coral'
    silence_seconds: float = Field(default=1.3, ge=0.2, le=5)
    vad_threshold: float = Field(default=0.015, gt=0, lt=1)
    vad_release_ratio: float = Field(default=0.5, gt=0, le=1)
    min_speech_seconds: float = Field(default=0.2, ge=0.1, le=2)
    max_turn_seconds: float = Field(default=60, ge=3, le=180)
    draft_delay: float = Field(default=0.6, ge=0, le=5)
    speculative_enabled: bool = True
    request_timeout: float = Field(default=30, gt=0, le=180)
    tool_timeout: float = Field(default=5, gt=0, le=60)
    max_tool_calls: int = Field(default=4, ge=1, le=16)
    history_turns: int = Field(default=10, ge=1, le=50)
    input_device: int | None = None
    output_device: int | None = None

    @classmethod
    def from_env(cls):
        # No implicit .env loading: secrets stay in the process environment.
        values = {}
        for name in cls.model_fields:
            if (value := os.getenv(name.upper())) is not None:
                values[name] = value
        return cls(**values)
