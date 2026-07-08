"""Microphone capture using sounddevice, with a scipy-based WAV writer.

The recorder runs an input stream; audio frames arrive on PortAudio's
internal thread and are buffered. On stop() we concatenate the buffer and
write a temporary 16-bit WAV via scipy, which Faster-Whisper reads directly.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample_poly


class AudioRecordingError(RuntimeError):
    """Raised when the microphone cannot be opened or recording fails."""


class AudioRecorder:
    """Records from the default input device into an in-memory buffer."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate  # TARGET rate handed to Whisper (16 kHz)
        self.channels = channels
        self.device = device  # None = system default; int = device index
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self.is_recording = False
        # The rate/channels we actually open the device at (may differ from the
        # target: hardware devices often only accept their native 44.1/48 kHz).
        self._capture_rate: int = sample_rate
        self._capture_channels: int = channels

    # -- callbacks ----------------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        """Called by PortAudio for every audio block. Keep it lightweight."""
        if status:
            # Over/underflows are non-fatal; just note them.
            print(f"[audio] stream status: {status}")
        with self._lock:
            self._frames.append(indata.copy())

    # -- control ------------------------------------------------------------
    def start(self) -> None:
        """Open the default microphone and begin buffering audio.

        Raises:
            AudioRecordingError: if no input device is available / mic fails.
        """
        if self.is_recording:
            return
        with self._lock:
            self._frames = []

        # Build a list of (rate, channels) attempts. Prefer the target 16 kHz
        # mono, but fall back to the device's native settings, because raw
        # hardware (hw:x,y) devices reject rates/channel-counts they don't
        # natively support. Whatever we capture gets resampled in stop().
        attempts: list[tuple[int, int]] = [(self.sample_rate, self.channels)]
        try:
            info = sd.query_devices(self.device, "input")
            native_rate = int(info["default_samplerate"])
            native_ch = max(1, int(info["max_input_channels"]))
            attempts += [
                (native_rate, 1),
                (native_rate, native_ch),
            ]
        except Exception:
            pass  # if we can't query, just try the target settings

        last_exc: Exception | None = None
        for rate, ch in attempts:
            try:
                self._stream = sd.InputStream(
                    samplerate=rate,
                    channels=ch,
                    device=self.device,
                    dtype="float32",
                    callback=self._callback,
                )
                self._stream.start()
                self._capture_rate = rate
                self._capture_channels = ch
                print(f"[audio] recording at {rate} Hz, {ch} ch (device={self.device})")
                self.is_recording = True
                return
            except Exception as exc:  # PortAudio rejects this combination
                last_exc = exc
                self._stream = None

        raise AudioRecordingError(
            f"Could not start microphone: {last_exc}. "
            "Check that a microphone is connected and not in use."
        )

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a float32 array.

        Raises:
            AudioRecordingError: if nothing was recorded.
        """
        if not self.is_recording or self._stream is None:
            raise AudioRecordingError("stop() called while not recording.")
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            self.is_recording = False

        with self._lock:
            frames = list(self._frames)
            self._frames = []

        if not frames:
            raise AudioRecordingError("No audio captured (empty recording).")

        audio = np.concatenate(frames, axis=0)

        # 1. Downmix any multi-channel capture to mono by averaging channels.
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 2. Resample from the capture rate to the 16 kHz Whisper expects.
        if self._capture_rate != self.sample_rate:
            audio = resample_poly(audio, self.sample_rate, self._capture_rate)

        return audio.astype(np.float32)

    # -- helpers ------------------------------------------------------------
    def save_wav(self, audio: np.ndarray) -> Path:
        """Write float32 audio to a temporary 16-bit PCM WAV (via scipy).

        Returns the path to the temp file. Caller is responsible for cleanup.
        The clip to [-1, 1] is important: loud input can exceed the float range
        and out-of-range samples make Whisper hallucinate.
        """
        # Convert float32 [-1, 1] to int16 for a standard PCM WAV.
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(
            prefix="whisperflow_", suffix=".wav", delete=False
        )
        tmp.close()
        wavfile.write(tmp.name, self.sample_rate, pcm16)
        return Path(tmp.name)

    def duration(self, audio: np.ndarray) -> float:
        """Length of the recording in seconds."""
        return len(audio) / float(self.sample_rate)

    @staticmethod
    def describe_default_input() -> str:
        """Return a human-readable description of the default input device."""
        try:
            default_in = sd.default.device[0]
            info = sd.query_devices(default_in, "input")
            return f"{info['name']} (index {default_in})"
        except Exception as exc:  # no input device at all
            return f"<no default input device: {exc}>"
