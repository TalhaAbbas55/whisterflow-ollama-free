"""Local speech-to-text using Faster-Whisper."""

from __future__ import annotations

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
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.initial_prompt = initial_prompt
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
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to load Whisper model '{self.model_size}': {exc}"
            ) from exc

    def transcribe(self, wav_path: str) -> str:
        """Transcribe a WAV file to text.

        Raises:
            TranscriptionError: on any failure during transcription.
        """
        if self._model is None:
            self.load()
        assert self._model is not None  # for type checkers

        try:
            segments, _info = self._model.transcribe(
                wav_path,
                language=self.language,
                initial_prompt=self.initial_prompt,
                beam_size=5,
                vad_filter=True,  # skip silence for cleaner output
            )
            text = " ".join(segment.text.strip() for segment in segments)
            return text.strip()
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
