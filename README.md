# WhisperFlow

A local, privacy-friendly **WhisperFlow alternative**. It lives in your system
tray, listens for a global hotkey, records your voice, transcribes it **locally**
with Faster-Whisper, rewrites it into a clean professional AI prompt using a
**local Ollama** model of your choice, and copies the result to your clipboard.

Everything runs on your machine — no audio or text leaves your computer.

## How it works

1. **Press `Ctrl + Shift + Z`** → a small floating microphone overlay appears and
   recording starts.
2. **Press `Ctrl + Shift + Z` again** → recording stops, then the app:
   - transcribes the audio locally (Faster-Whisper),
   - sends the transcript to the Ollama model you selected to rewrite it as a polished prompt,
   - copies the enhanced prompt to the clipboard,
   - hides the overlay and shows a "Prompt copied!" notification.

Extras included: last-10-prompts history, a system tray icon, and optional
"start with Windows".

## Project layout

```
main.py                 # entry point
requirements.txt
whisperflow/
  config.py             # all settings (hotkey, model, etc.)
  audio.py              # microphone capture (sounddevice) + WAV writer (scipy)
  transcriber.py        # Faster-Whisper speech-to-text
  enhancer.py           # Ollama prompt rewriting
  overlay.py            # floating mic overlay (tkinter)
  notifier.py           # toast notifications (tkinter)
  history.py            # last-10 prompts, persisted to JSON
  tray.py               # system tray icon (pystray)
  startup.py            # optional Windows autostart (registry)
  app.py                # orchestrator / state machine
```

## Prerequisites

- **Python 3.12+**
- **Ollama** installed and running, with at least one model pulled. Any chat
  model works; the app lists whatever you have and lets you pick one at startup.
  ```bash
  ollama serve                    # if not already running as a service
  ollama pull <model>             # e.g. ollama pull llama3.2:3b
  ollama list                     # see the models you already have
  ```
- A working **microphone**.

## Installation

### Option A — with uv (recommended, this repo is uv-managed)

```bash
uv sync                  # installs everything from pyproject.toml / uv.lock
uv run python main.py
```

### Option B — with pip + venv

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

### tkinter note

`tkinter` is part of Python's standard library and **cannot be installed via
pip** (so `uv add tkinter` / `pip install tkinter` will fail — that's expected).

- **Windows / macOS**: it's bundled with the official python.org installer. ✔
- **Linux**: install it via your package manager, e.g.
  `sudo apt install python3-tk`.

## Usage

1. Make sure Ollama is running and you have at least one model pulled.
2. Start the app:
   ```bash
   uv run python main.py     # or: python main.py
   ```
   On startup it lists the Ollama models you have installed and asks which one to
   use. Just enter the **number** of the model and press Enter; an invalid number
   shows an error and re-lists your options. The first launch also downloads the
   Whisper model (a few hundred MB) once.
3. Press **`Ctrl + Shift + Z`** to start recording, speak, then press it again
   to stop. The enhanced prompt is now on your clipboard — paste it anywhere.
4. Right-click the tray icon for **Show history**, **Start with Windows**, and **Quit**.

## Configuration

Edit `whisperflow/config.py`:

| Setting | Default | Notes |
|---|---|---|
| `hotkey` | `ctrl+shift+z` | e.g. `ctrl+shift+z`, `ctrl+alt+d` |
| `whisper_model` | `small` | `tiny`/`base`/`small`/`medium`/`large-v3` (bigger = more accurate, slower) |
| `whisper_device` | `cpu` | `cuda` for NVIDIA GPU |
| `whisper_compute_type` | `int8` | `float16` for GPU |
| `whisper_language` | `en` | force a language, or `None` to auto-detect |
| `whisper_initial_prompt` | _(dev vocabulary)_ | context hint that biases transcription toward your usual vocabulary |
| `ollama_model` | _(chosen at startup)_ | You pick from your installed models each run; this value is only a fallback for non-interactive startup (e.g. Windows autostart) |
| `ollama_host` | `http://localhost:11434` | local Ollama server URL |
| `history_size` | `10` | prompts kept in `~/.whisperflow_history.json` |

## Notes & troubleshooting

- **Global hotkey**: no root/sudo needed. On **Linux** keys are read from
  `/dev/input` via `evdev`, which works on both Wayland and X11 as long as your
  user is in the `input` group:
  ```bash
  sudo usermod -aG input $USER   # then log out and back in
  ```
  On **Windows/macOS** the hotkey uses `pynput` and works out of the box. (On a
  native Wayland session without `evdev` access, the `pynput` fallback only
  fires while an X11/XWayland window is focused.)
- **Wrong / garbled transcript**: use a bigger `whisper_model` (`small` →
  `medium`), set `whisper_language` to your language, and tailor
  `whisper_initial_prompt` to the words you commonly dictate.
- **Ollama errors** (`Could not reach Ollama…`): start `ollama serve`. If you have
  no models installed, pull one (e.g. `ollama pull llama3.2:3b`) and restart.
- **No microphone / mic in use**: the app shows an error toast and stays running;
  fix the device and try again.
- **No tray icon**: the app still works fully; tray just requires `pystray` +
  `pillow`.
- Errors during transcription/enhancement are caught and shown as a toast — the
  app never crashes on a single failed recording.

## Privacy

Audio is processed locally by Faster-Whisper and never uploaded. Text is sent
only to your **local** Ollama server. The temporary WAV file is deleted after
each transcription.
