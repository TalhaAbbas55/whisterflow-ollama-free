"""WhisperFlowApp: the orchestrator wiring hotkey -> record -> transcribe ->
enhance -> clipboard together.

Threading model
---------------
- The Tk root and its mainloop live on the MAIN thread. Every GUI mutation is
  marshalled there via `root.after(0, ...)`.
- The global hotkey callback fires on the hotkey listener's thread
  (evdev on Linux, pynput elsewhere — see whisperflow/hotkey.py).
- Heavy work (Whisper + Ollama) runs on a short-lived worker thread so the UI
  never freezes.

State machine: IDLE -> RECORDING -> PROCESSING -> IDLE
"""



from __future__ import annotations

import enum
import threading
import tkinter as tk

import pyperclip

from .audio import AudioRecorder, AudioRecordingError
from .config import Config
from .enhancer import EnhancementError, PromptEnhancer
from .history import HistoryStore
from .hotkey import HotkeyError, create_hotkey_listener
from .notifier import Toast
from .overlay import MicOverlay
from .transcriber import Transcriber, TranscriptionError

try:
    from . import startup
    from .tray import TrayIcon

    _TRAY_AVAILABLE = True
except Exception:  # pystray/PIL missing -> run without a tray
    _TRAY_AVAILABLE = False


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class WhisperFlowApp:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.state = State.IDLE
        self._state_lock = threading.Lock()

        # Core components.
        self.recorder = AudioRecorder(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            device=self.config.input_device,
        )
        self.transcriber = Transcriber(
            model_size=self.config.whisper_model,
            device=self.config.whisper_device,
            compute_type=self.config.whisper_compute_type,
            language=self.config.whisper_language,
            initial_prompt=self.config.whisper_initial_prompt,
            beam_size=self.config.whisper_beam_size,
            cpu_threads=self.config.whisper_cpu_threads,
        )
        self.enhancer = PromptEnhancer(
            model=self.config.ollama_model,
            host=self.config.ollama_host,
            timeout=self.config.ollama_timeout,
            keep_alive=self.config.ollama_keep_alive,
        )
        self.history = HistoryStore(
            self.config.history_file, max_items=self.config.history_size
        )

        # GUI (created on the main thread in run()).
        self.root: tk.Tk | None = None
        self.overlay: MicOverlay | None = None
        self.toast: Toast | None = None
        self.tray: "TrayIcon | None" = None
        self._hotkey_listener = None  # created in run() via create_hotkey_listener

    # -- GUI marshalling ----------------------------------------------------
    def _ui(self, func, *args) -> None:
        """Schedule `func(*args)` on the Tk main thread."""
        if self.root is not None:
            self.root.after(0, lambda: func(*args))

    # -- hotkey -------------------------------------------------------------
    def _on_hotkey(self) -> None:
        """Toggle handler. Runs on the hotkey listener's thread."""
        with self._state_lock:
            current = self.state

        if current == State.IDLE:
            self._begin_recording()
        elif current == State.RECORDING:
            self._end_recording()
        else:
            # PROCESSING: ignore presses until the previous job finishes.
            print("[app] busy processing; ignoring hotkey")

    # -- recording ----------------------------------------------------------
    def _begin_recording(self) -> None:
        try:
            self.recorder.start()
        except AudioRecordingError as exc:
            self._ui(self._toast_error, str(exc))
            return
        with self._state_lock:
            self.state = State.RECORDING
        self._ui(self.overlay.show_recording)

    def _end_recording(self) -> None:
        with self._state_lock:
            self.state = State.PROCESSING
        self._ui(self.overlay.show_processing)
        # Do the heavy lifting off the GUI/hotkey thread.
        threading.Thread(target=self._process_recording, daemon=True).start()

    def _process_recording(self) -> None:
        """Stop audio, transcribe, enhance, copy. Runs on a worker thread."""
        wav_path = None
        try:
            audio = self.recorder.stop()
            dur = self.recorder.duration(audio)
            # Peak amplitude tells us whether the mic actually captured sound.
            # Near-zero peak = recording silence (often the sudo/PulseAudio issue).
            peak = float(abs(audio).max()) if audio.size else 0.0
            print(f"[app] captured {dur:.1f}s, peak amplitude={peak:.4f}")
            if dur < 0.3:
                raise AudioRecordingError("Recording too short — try again.")
            if peak < 0.005:
                raise AudioRecordingError(
                    "Microphone captured only silence (peak≈0). "
                    "On Linux, running with sudo can block access to your mic. "
                    "Check the input device / try without sudo."
                )

            # Write a clipped, normalised 16-bit WAV, then transcribe it. The
            # clip in save_wav keeps out-of-range samples from making Whisper
            # hallucinate.
            wav_path = self.recorder.save_wav(audio)

            transcript = self.transcriber.transcribe(str(wav_path))
            print(f"[app] transcript: {transcript!r}")
            if not transcript:
                raise TranscriptionError("No speech detected.")

            enhanced = self.enhancer.enhance(transcript)
            print(f"[app] enhanced:   {enhanced!r}")

            pyperclip.copy(enhanced)
            self.history.add(transcript, enhanced)

            preview = (enhanced[:80] + "…") if len(enhanced) > 80 else enhanced
            self._ui(self._toast_success, f"✅ Prompt copied!\n{preview}")

        except (AudioRecordingError, TranscriptionError, EnhancementError) as exc:
            self._ui(self._toast_error, str(exc))
        except Exception as exc:  # last-resort safety net
            self._ui(self._toast_error, f"Unexpected error: {exc}")
        finally:
            # Clean up temp WAV and reset state.
            if wav_path is not None:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._ui(self.overlay.hide)
            with self._state_lock:
                self.state = State.IDLE

    # -- toasts (must run on GUI thread) ------------------------------------
    def _toast_success(self, msg: str) -> None:
        if self.toast:
            self.toast.show(msg, duration_ms=3000)
        if self.tray:
            self.tray.notify(msg, self.config.app_name)

    def _toast_error(self, msg: str) -> None:
        if self.overlay:
            self.overlay.hide()
        if self.toast:
            self.toast.show(f"⚠ {msg}", duration_ms=4000, error=True)

    # -- history viewer -----------------------------------------------------
    def show_history(self) -> None:
        """Open a window listing recent prompts. Marshalled to GUI thread."""
        self._ui(self._build_history_window)

    def _build_history_window(self) -> None:
        assert self.root is not None
        win = tk.Toplevel(self.root)
        win.title(f"{self.config.app_name} — History")
        win.geometry("560x460")
        win.configure(bg="#1e1e2e")

        txt = tk.Text(win, wrap="word", bg="#181825", fg="#cdd6f4", padx=12, pady=12)
        txt.pack(fill="both", expand=True)

        items = self.history.items()
        if not items:
            txt.insert("end", "No history yet.")
        else:
            for i, entry in enumerate(reversed(items), 1):
                txt.insert("end", f"#{i}  {entry['timestamp']}\n")
                txt.insert("end", f"  spoken:   {entry['original']}\n")
                txt.insert("end", f"  enhanced: {entry['enhanced']}\n\n")
        txt.config(state="disabled")

    # -- autostart toggle ---------------------------------------------------
    def _toggle_autostart(self) -> None:
        if not _TRAY_AVAILABLE:
            return
        if startup.is_enabled():
            startup.disable()
        else:
            startup.enable()

    # -- lifecycle ----------------------------------------------------------
    def run(self) -> None:
        """Set up GUI, hotkey and tray, then block on the Tk mainloop."""
        # 1. Tk root lives here, hidden — it owns overlay + toasts.
        self.root = tk.Tk()
        self.root.withdraw()  # we only use Toplevels
        self.overlay = MicOverlay(self.root)
        self.toast = Toast(self.root)

        # Show which microphone we'll record from (helps diagnose silent audio).
        print(f"[app] default input device: {self.recorder.describe_default_input()}")

        # 2. Pre-load + warm up both models so the FIRST dictation isn't slow.
        #    - Whisper: load weights, then run one throwaway decode (the first
        #      real decode is otherwise noticeably slower).
        #    - Ollama: preload the model into memory in the background so it's
        #      warm by the time the user finishes speaking, and keep it warm.
        print(f"[app] loading Whisper model '{self.config.whisper_model}'…")
        try:
            self.transcriber.load()
        except TranscriptionError as exc:
            print(f"[app] WARNING: {exc}")
        threading.Thread(target=self.transcriber.warm_up, daemon=True).start()
        threading.Thread(target=self.enhancer.warm_up, daemon=True).start()

        # 3. System tray (optional).
        if _TRAY_AVAILABLE:
            self.tray = TrayIcon(
                app_name=self.config.app_name,
                on_quit=self.quit,
                on_show_history=self.show_history,
                on_toggle_autostart=self._toggle_autostart,
                autostart_enabled=startup.is_enabled,
                autostart_supported=startup.is_supported(),
            )
            self.tray.start()

        # 4. Register the global hotkey (listener runs on its own thread).
        try:
            self._hotkey_listener = create_hotkey_listener(
                self.config.hotkey, self._on_hotkey
            )
            self._hotkey_listener.start()
        except HotkeyError as exc:
            print(f"[app] WARNING: global hotkey unavailable: {exc}")

        print(
            f"[app] {self.config.app_name} ready. "
            f"Press {self.config.hotkey.upper()} to start/stop recording."
        )
        if self.toast:
            self.toast.show(
                f"{self.config.app_name} running\nPress "
                f"{self.config.hotkey.upper()} to record",
                duration_ms=3500,
            )

        # 5. Block on the GUI event loop (main thread).
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit()

    def quit(self) -> None:
        """Tear everything down cleanly."""
        print("[app] shutting down…")
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:
                pass
            self._hotkey_listener = None
        if self.recorder.is_recording:
            try:
                self.recorder.stop()
            except AudioRecordingError:
                pass
        if self.tray:
            self.tray.stop()
        if self.root is not None:
            # Must quit the loop from the GUI thread.
            self.root.after(0, self.root.destroy)
