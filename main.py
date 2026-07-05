"""WhisperFlow entry point.

Run with:
    python main.py   # Linux, macOS and Windows, no root/sudo required

On Linux the global hotkey is read from /dev/input via evdev, which works on
both Wayland and X11 without root (your user must be in the 'input' group).
Other platforms use pynput.
"""

from __future__ import annotations

import sys

from whisperflow.app import WhisperFlowApp
from whisperflow.config import Config
from whisperflow.enhancer import EnhancementError, list_installed_models


def choose_model(config: Config) -> str | None:
    """Ask the user which installed Ollama model to use.

    Lists the models currently installed on the local Ollama server, then loops
    until the user enters the name (or list number) of one that actually exists.
    Returns the chosen model name, or None if selection cannot proceed.
    """
    # If there's no interactive console (e.g. Windows autostart with pythonw),
    # skip the prompt and fall back to the configured default.
    if not sys.stdin or not sys.stdin.isatty():
        print(f"[startup] no interactive console; using default model "
              f"'{config.ollama_model}'.")
        return config.ollama_model

    # 1. Fetch the installed models (fails clearly if Ollama isn't running).
    try:
        models = list_installed_models(config.ollama_host)
    except EnhancementError as exc:
        print(f"[startup] {exc}", file=sys.stderr)
        return None

    if not models:
        print(
            "[startup] No Ollama models are installed. Pull one first, e.g.:\n"
            "    ollama pull llama3.2:3b",
            file=sys.stderr,
        )
        return None

    # 2. Loop until the user enters the NUMBER of a valid model.
    while True:
        print("\nInstalled Ollama models:")
        for i, name in enumerate(models, 1):
            print(f"  {i}. {name}")

        choice = input(
            f"\nEnter the number of the model to use (1-{len(models)}) and press Enter: "
        ).strip()
        if not choice:
            continue

        if not choice.isdigit():
            print(f"  ✗ Please enter a number between 1 and {len(models)}.")
            continue

        idx = int(choice)
        if 1 <= idx <= len(models):
            return models[idx - 1]
        print(f"  ✗ '{choice}' is out of range. Enter a number between 1 and {len(models)}.")


def main() -> int:
    # Start from defaults, then let the user pick the Ollama model at runtime.
    config = Config()

    model = choose_model(config)
    if model is None:
        return 1  # could not select a model (Ollama down or none installed)
    config.ollama_model = model
    print(f"[startup] using Ollama model: {config.ollama_model}")

    app = WhisperFlowApp(config)
    try:
        app.run()
    except Exception as exc:  # surface fatal startup errors to the console
        print(f"[fatal] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


