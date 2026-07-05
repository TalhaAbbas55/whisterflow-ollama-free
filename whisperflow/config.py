"""Central, editable configuration for WhisperFlow.

All tunable knobs live here so the rest of the code stays clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- Global hotkey -----------------------------------------------------
    # Uses the `keyboard` library syntax. Toggle: press once to start, again to stop.
    hotkey: str = "ctrl+shift+z"

    # --- Audio capture -----------------------------------------------------
    # Whisper expects 16 kHz mono audio; recording at that rate avoids resampling.
    sample_rate: int = 16000
    channels: int = 1
    # Input device for recording. None = system default. On Linux the default
    # is often ALSA's generic "default" which may capture silence; set this to
    # the integer index of your real mic (run tools/list_devices.py to find it).
    input_device: int | str | None = 5

    # --- Faster-Whisper (local transcription) ------------------------------
    # model_size: tiny | base | small | medium | large-v3
    #   - "small" is noticeably more accurate than "base" (especially with
    #     accented speech) and still fast enough on a modern CPU.
    # device: "cpu" or "cuda" (if you have an NVIDIA GPU + CUDA installed).
    # compute_type: "int8" (fast on CPU), "float16" (GPU), "int8_float16", etc.
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # None = auto-detect spoken language. Set e.g. "en" to force English.
    # Forcing it avoids wrong-language guesses on short clips.
    whisper_language: str | None = "en"
    # Fed to Whisper as context before decoding; biases it toward this
    # vocabulary/style. Tune it to what you usually dictate.
    whisper_initial_prompt: str | None = (
        "A software developer dictating a programming task: Python, function, "
        "variable, API, even or odd, prime number, leap year, string, list."
    )

    # --- Ollama (local LLM prompt enhancement) -----------------------------
    ollama_model: str = "qwen2.5-coder:latest"
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: float = 120.0  # seconds

    # --- History -----------------------------------------------------------
    history_size: int = 10
    history_file: Path = field(
        default_factory=lambda: Path.home() / ".whisperflow_history.json"
    )

    # --- Misc --------------------------------------------------------------
    app_name: str = "WhisperFlow"



