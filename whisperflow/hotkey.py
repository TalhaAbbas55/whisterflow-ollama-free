"""Global hotkey listening with a Wayland-safe Linux backend.

Two backends, picked automatically by `create_hotkey_listener`:

- EvdevHotkeyListener (Linux): reads key events straight from
  /dev/input/event* devices, below the display server, so it works on both
  Wayland and X11 without root. Requires read access to the input devices
  (being in the `input` group is the usual way).
- PynputHotkeyListener (fallback / Windows / macOS / X11): registers the
  combo through pynput's GlobalHotKeys. On a native Wayland session this
  only fires while an X11/XWayland window is focused, which is why evdev
  is preferred on Linux.

Both expose the same tiny interface: start() / stop(). The callback runs on
the listener's own thread; callers must marshal GUI work themselves.
"""

from __future__ import annotations

import selectors
import sys
import threading
import time


class HotkeyError(Exception):
    """Raised when a hotkey backend cannot be set up."""


# Aliases normalised before mapping to a backend's key names.
_ALIASES = {
    "control": "ctrl",
    "win": "cmd",
    "super": "cmd",
    "meta": "cmd",
    "option": "alt",
    "return": "enter",
    "esc": "escape",
}

_MODIFIERS = {"ctrl", "shift", "alt", "cmd"}


def _parse_spec(hotkey: str) -> tuple[set[str], str]:
    """Split "ctrl+shift+z" into ({"ctrl", "shift"}, "z")."""
    mods: set[str] = set()
    key: str | None = None
    for raw in hotkey.split("+"):
        part = raw.strip().lower()
        if not part:
            continue
        part = _ALIASES.get(part, part)
        if part in _MODIFIERS:
            mods.add(part)
        elif key is None:
            key = part
        else:
            raise HotkeyError(f"Hotkey {hotkey!r} has more than one non-modifier key.")
    if key is None:
        raise HotkeyError(f"Hotkey {hotkey!r} has no non-modifier key.")
    return mods, key


# -- evdev backend (Linux, Wayland-safe) -------------------------------------
class EvdevHotkeyListener:
    """Fires `callback` when the combo is pressed on any keyboard device."""

    # Seconds within which a repeated fire is ignored. Guards against the
    # same combo arriving from two devices (e.g. a physical keyboard plus a
    # remapper's virtual one).
    _DEBOUNCE_S = 0.3

    def __init__(self, hotkey: str, callback) -> None:
        import evdev  # imported lazily so non-Linux platforms never need it

        self._evdev = evdev
        self._callback = callback
        self._thread: threading.Thread | None = None
        self._stop_r, self._stop_w = None, None
        self._last_fire = 0.0

        mods, key = _parse_spec(hotkey)
        e = evdev.ecodes
        mod_groups = {
            "ctrl": {e.KEY_LEFTCTRL, e.KEY_RIGHTCTRL},
            "shift": {e.KEY_LEFTSHIFT, e.KEY_RIGHTSHIFT},
            "alt": {e.KEY_LEFTALT, e.KEY_RIGHTALT},
            "cmd": {e.KEY_LEFTMETA, e.KEY_RIGHTMETA},
        }
        self._mod_groups = [mod_groups[m] for m in mods]
        self._target = self._key_to_code(key)

        self._devices = self._find_keyboards()
        if not self._devices:
            raise HotkeyError(
                "No readable keyboard devices under /dev/input. "
                "Add your user to the 'input' group and log out/in:\n"
                "    sudo usermod -aG input $USER"
            )

    def _key_to_code(self, key: str) -> int:
        e = self._evdev.ecodes
        special = {
            "space": e.KEY_SPACE,
            "enter": e.KEY_ENTER,
            "tab": e.KEY_TAB,
            "escape": e.KEY_ESC,
            "backspace": e.KEY_BACKSPACE,
            "delete": e.KEY_DELETE,
            "insert": e.KEY_INSERT,
            "home": e.KEY_HOME,
            "end": e.KEY_END,
            "pageup": e.KEY_PAGEUP,
            "pagedown": e.KEY_PAGEDOWN,
            "up": e.KEY_UP,
            "down": e.KEY_DOWN,
            "left": e.KEY_LEFT,
            "right": e.KEY_RIGHT,
        }
        if key in special:
            return special[key]
        code = getattr(e, f"KEY_{key.upper()}", None)
        if code is None:
            raise HotkeyError(f"Unsupported hotkey key: {key!r}")
        return code

    def _find_keyboards(self) -> list:
        """Open every device that looks like a keyboard (has A + LeftCtrl)."""
        e = self._evdev.ecodes
        devices = []
        for path in self._evdev.list_devices():
            try:
                dev = self._evdev.InputDevice(path)
                keys = dev.capabilities().get(e.EV_KEY, [])
                if e.KEY_A in keys and e.KEY_LEFTCTRL in keys:
                    devices.append(dev)
                else:
                    dev.close()
            except OSError:
                continue
        return devices

    def start(self) -> None:
        import os

        self._stop_r, self._stop_w = os.pipe()
        self._thread = threading.Thread(
            target=self._loop, name="evdev-hotkey", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        e = self._evdev.ecodes
        sel = selectors.DefaultSelector()
        sel.register(self._stop_r, selectors.EVENT_READ, data=None)
        for dev in self._devices:
            sel.register(dev.fd, selectors.EVENT_READ, data=dev)

        pressed: set[int] = set()
        running = True
        while running:
            for key, _ in sel.select():
                if key.data is None:  # stop pipe
                    running = False
                    break
                dev = key.data
                try:
                    events = list(dev.read())
                except OSError:  # device unplugged
                    sel.unregister(dev.fd)
                    dev.close()
                    continue
                for ev in events:
                    if ev.type != e.EV_KEY:
                        continue
                    if ev.value == 1:  # key down
                        pressed.add(ev.code)
                        if ev.code == self._target and all(
                            pressed & group for group in self._mod_groups
                        ):
                            now = time.monotonic()
                            if now - self._last_fire >= self._DEBOUNCE_S:
                                self._last_fire = now
                                self._callback()
                    elif ev.value == 0:  # key up
                        pressed.discard(ev.code)

        sel.close()
        for dev in self._devices:
            try:
                dev.close()
            except OSError:
                pass

    def stop(self) -> None:
        import os

        if self._stop_w is not None:
            try:
                os.write(self._stop_w, b"x")
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        for fd in (self._stop_r, self._stop_w):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._stop_r = self._stop_w = None


# -- pynput backend (X11 / Windows / macOS) ----------------------------------
class PynputHotkeyListener:
    def __init__(self, hotkey: str, callback) -> None:
        from pynput import keyboard as pk

        mods, key = _parse_spec(hotkey)
        # pynput wraps named keys in angle brackets; bare chars stay bare.
        parts = [f"<{m}>" for m in sorted(mods)]
        parts.append(key if len(key) == 1 else f"<{key}>")
        self._listener = pk.GlobalHotKeys({"+".join(parts): callback})

    def start(self) -> None:
        self._listener.start()
        self._listener.wait()

    def stop(self) -> None:
        self._listener.stop()


def create_hotkey_listener(hotkey: str, callback):
    """Pick the best backend for this platform.

    Linux prefers evdev (works on Wayland, no root); anything else, or a
    Linux box without /dev/input access, falls back to pynput.
    """
    if sys.platform.startswith("linux"):
        try:
            listener = EvdevHotkeyListener(hotkey, callback)
            print("[hotkey] using evdev backend (Wayland-safe)")
            return listener
        except (ImportError, HotkeyError) as exc:
            print(f"[hotkey] evdev backend unavailable ({exc}); using pynput")
    listener = PynputHotkeyListener(hotkey, callback)
    print("[hotkey] using pynput backend")
    return listener
