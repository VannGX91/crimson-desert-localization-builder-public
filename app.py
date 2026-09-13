import importlib
import sys

from crimson0020.native_checksum import native_checksum_available
from crimson0020.webview_app import run_app, validate_web_assets


def run_smoke_test() -> None:
    for module_name in ("cryptography", "lz4", "webview", "clr"):
        importlib.import_module(module_name)
    validate_web_assets()
    if not native_checksum_available():
        raise RuntimeError("Native checksum library is unavailable.")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--smoke-test"]:
        run_smoke_test()
    else:
        run_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
