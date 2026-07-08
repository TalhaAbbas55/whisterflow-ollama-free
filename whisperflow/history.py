"""Lightweight on-disk history of the last N enhanced prompts."""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path


def _is_writable(path: Path) -> bool:
    """True if we can write `path` (existing file writable, or parent writable)."""
    if path.exists():
        return os.access(path, os.W_OK)
    parent = path.parent
    return parent.exists() and os.access(parent, os.W_OK)


def _resolve_writable_path(preferred: Path) -> Path:
    """Return the preferred path if writable, else a per-user fallback.

    The usual failure is a history file left owned by root from a past `sudo`
    run: the configured path then exists but isn't writable by the normal user.
    Rather than silently losing history, fall back to the XDG state directory,
    and finally to a temp file, so persistence keeps working.
    """
    if _is_writable(preferred):
        return preferred

    state_home = os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")
    fallback = Path(state_home) / "whisperflow" / "history.json"
    try:
        fallback.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    if _is_writable(fallback):
        print(
            f"[history] {preferred} is not writable (owned by another user?); "
            f"using {fallback} instead."
        )
        return fallback

    temp = Path(tempfile.gettempdir()) / "whisperflow_history.json"
    print(
        f"[history] {preferred} is not writable; falling back to {temp} "
        "(cleared on reboot)."
    )
    return temp


class HistoryStore:
    """Keeps the most recent prompts, persisted to a JSON file."""

    def __init__(self, path: Path, max_items: int = 10) -> None:
        self.path = _resolve_writable_path(Path(path))
        self.max_items = max_items
        self._items: deque[dict] = deque(maxlen=max_items)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for entry in data[-self.max_items :]:
                self._items.append(entry)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[history] could not read {self.path}: {exc}")

    def add(self, original: str, enhanced: str) -> None:
        """Record a new (original, enhanced) pair and persist it."""
        self._items.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "original": original,
                "enhanced": enhanced,
            }
        )
        self._save()

    def _save(self) -> None:
        try:
            self.path.write_text(
                json.dumps(list(self._items), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"[history] could not write {self.path}: {exc}")

    def items(self) -> list[dict]:
        """Return history newest-last."""
        return list(self._items)
