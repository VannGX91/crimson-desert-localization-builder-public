import ctypes
import os
import sys
from pathlib import Path


LAUNCHER_EXE_NAME = "Crimson.Desert.Localization.Builder.exe"
LAUNCHER_DIRECTORY_ENV = "CRIMSON_LAUNCHER_DIRECTORY"
SMOKE_TEST_ENV = "CRIMSON_SMOKE_TEST"


def _show_bootstrap_error(error: BaseException) -> None:
    message = f"The localization builder could not start.\n\n{error}"
    try:
        ctypes.windll.user32.MessageBoxW(
            None,
            message,
            "Crimson Desert Localization Builder",
            0x10,
        )
    except Exception:
        pass


def _bootstrap_release() -> None:
    executable = Path(sys.executable).resolve()
    if executable.name.casefold() != LAUNCHER_EXE_NAME.casefold():
        return

    root = executable.parent
    runtime = root / "_internal" / "runtime"
    os.environ[LAUNCHER_DIRECTORY_ENV] = str(root)
    os.environ["TCL_LIBRARY"] = str(runtime / "tcl" / "tcl8.6")
    os.environ["TK_LIBRARY"] = str(runtime / "tcl" / "tk8.6")
    smoke_test = os.environ.pop(SMOKE_TEST_ENV, "") == "1"
    try:
        from app import main

        exit_code = main(["--smoke-test"] if smoke_test else [])
    except BaseException as error:
        _show_bootstrap_error(error)
        os._exit(1)
    os._exit(int(exit_code or 0))


_bootstrap_release()
