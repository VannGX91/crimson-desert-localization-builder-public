import argparse
import platform
import subprocess
import sys
from pathlib import Path


def compiler_command(
    source: Path,
    output: Path,
    executable: str = sys.executable,
) -> tuple[str, ...]:
    return (
        executable,
        "-m",
        "ziglang",
        "cc",
        "-O3",
        "-s",
        "-shared",
        "-target",
        "x86_64-windows-gnu",
        "-o",
        str(output),
        str(source),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the native PA checksum helper.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if platform.system() != "Windows" or platform.machine().lower() not in {
        "amd64",
        "x86_64",
    }:
        raise SystemExit("The release native checksum build requires 64-bit Windows.")
    project = Path(__file__).resolve().parent
    source = project / "native" / "pa_checksum.c"
    output = (
        args.output.expanduser().resolve()
        if args.output
        else project / "crimson0020" / "native" / "pa_checksum.dll"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(compiler_command(source, output), check=True)
    print(f"Built native checksum library: {output}")


if __name__ == "__main__":
    main()
