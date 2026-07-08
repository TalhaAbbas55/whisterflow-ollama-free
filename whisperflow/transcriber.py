"""Local speech-to-text using Faster-Whisper."""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel


class TranscriptionError(RuntimeError):
    """Raised when transcription fails."""


class Transcriber:
    """Wraps a Faster-Whisper model. The model is loaded lazily on first use."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        initial_prompt: str | None = None,
        beam_size: int = 1,
        cpu_threads: int = 0,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.initial_prompt = initial_prompt
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self._model: WhisperModel | None = None

    def load(self) -> None:
        """Load (and on first run, download) the Whisper model weights."""
        if self._model is not None:
            return
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to load Whisper model '{self.model_size}': {exc}"
            ) from exc

    def warm_up(self) -> None:
        """Run one throwaway inference so the first real dictation isn't slow.

        Loading the weights (load()) doesn't pay the one-off cost of the first
        decode; doing it here on a short buffer of silence gets that out of the
        way at startup. Best-effort: failures are ignored.
        """
        try:
            self.load()
            assert self._model is not None
            silence = np.zeros(16000, dtype=np.float32)  # 1 s at 16 kHz
            segments, _ = self._model.transcribe(silence, beam_size=self.beam_size)
            list(segments)  # segments is a generator; force it to run
        except Exception:
            pass

    def transcribe(self, audio: np.ndarray | str) -> str:
        """Transcribe audio to text.

        `audio` may be a float32 mono 16 kHz numpy array (passed straight
        through, no disk I/O) or a path to a WAV file.

        Raises:
            TranscriptionError: on any failure during transcription.
        """
        if self._model is None:
            self.load()
        assert self._model is not None  # for type checkers

        try:
            segments, _info = self._model.transcribe(
                audio,
                language=self.language,
                initial_prompt=self.initial_prompt,
                beam_size=self.beam_size,
                vad_filter=True,  # skip silence for cleaner output
            )
            text = " ".join(segment.text.strip() for segment in segments)
            return text.strip()
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
