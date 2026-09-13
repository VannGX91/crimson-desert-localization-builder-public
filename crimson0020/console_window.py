"""Hide child-process console windows so the packaged app stays friendly.

Third-party helpers (notably the JavaScript runtime used by the ``translators``
engine) launch console programs such as ``node`` without suppressing their
window, which flashes a series of CMD windows.  The packaged runtime uses a
``._pth`` file without ``import site``, so ``sitecustomize`` is not loaded;
this module is imported through :mod:`crimson0020` instead.
"""

from __future__ import annotations

import os

CREATE_NO_WINDOW = 0x08000000
_INSTALLED = False


def no_window_creationflags(flags: int) -> int:
    """Return process creation flags with CREATE_NO_WINDOW set on Windows."""

    resolved = int(flags or 0)
    if os.name != "nt":
        return resolved
    return resolved | CREATE_NO_WINDOW


def install_no_window_subprocess() -> None:
    """Wrap ``subprocess.Popen`` once so every child hides its console."""

    global _INSTALLED
    if _INSTALLED or os.name != "nt":
        return
    import subprocess

    existing = subprocess.Popen
    if getattr(existing, "_crimson_no_window", False):
        _INSTALLED = True
        return

    class _NoWindowPopen(existing):  # type: ignore[misc]
        _crimson_no_window = True

        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["creationflags"] = no_window_creationflags(
                int(kwargs.get("creationflags") or 0)
            )
            super().__init__(*args, **kwargs)

    subprocess.Popen = _NoWindowPopen  # type: ignore[assignment]
    _INSTALLED = True
