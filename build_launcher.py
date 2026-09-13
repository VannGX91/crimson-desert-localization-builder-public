import os
import re
import shutil
import subprocess
from pathlib import Path


FRAMEWORK_ASSEMBLIES = (
    "mscorlib.dll",
    "System.dll",
    "System.Core.dll",
    "System.Windows.Forms.dll",
)


def launcher_compile_command(
    *,
    dotnet: Path,
    compiler: Path,
    framework: Path,
    source: Path,
    destination: Path,
    icon: Path,
) -> tuple[str, ...]:
    references = tuple(
        f"/reference:{framework / filename}" for filename in FRAMEWORK_ASSEMBLIES
    )
    return (
        str(dotnet),
        str(compiler),
        "/noconfig",
        "/nostdlib+",
        "/target:winexe",
        "/platform:x64",
        "/optimize+",
        "/deterministic+",
        "/debug-",
        "/utf8output",
        f"/out:{destination}",
        f"/win32icon:{icon}",
        *references,
        str(source),
    )


def _find_roslyn_compiler(dotnet: Path) -> Path:
    result = subprocess.run(
        (str(dotnet), "--list-sdks"),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    candidates: list[Path] = []
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"\s*(\S+)\s+\[(.+)]\s*", line)
        if match is None:
            continue
        version, sdk_root = match.groups()
        compiler = Path(sdk_root) / version / "Roslyn" / "bincore" / "csc.dll"
        if compiler.is_file():
            candidates.append(compiler)
    if not candidates:
        raise FileNotFoundError(
            "Microsoft Roslyn C# compiler was not found in any installed .NET SDK."
        )
    return candidates[-1]


def _find_framework_assemblies() -> Path:
    windows = Path(os.environ.get("WINDIR", r"C:\Windows"))
    framework = windows / "Microsoft.NET" / "Framework64" / "v4.0.30319"
    missing = [name for name in FRAMEWORK_ASSEMBLIES if not (framework / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required .NET Framework assemblies are missing: " + ", ".join(missing)
        )
    return framework


def compile_launcher(source: Path, destination: Path, icon: Path) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    icon = icon.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Launcher source is missing: {source}")
    if not icon.is_file():
        raise FileNotFoundError(f"Launcher icon is missing: {icon}")
    dotnet_command = shutil.which("dotnet")
    if dotnet_command is None:
        raise FileNotFoundError("Microsoft .NET SDK host was not found on PATH.")
    dotnet = Path(dotnet_command)
    compiler = _find_roslyn_compiler(dotnet)
    framework = _find_framework_assemblies()
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        launcher_compile_command(
            dotnet=dotnet,
            compiler=compiler,
            framework=framework,
            source=source,
            destination=destination,
            icon=icon,
        ),
        check=True,
    )
    if not destination.is_file():
        raise FileNotFoundError(f"Launcher compiler produced no executable: {destination}")
    return destination


if __name__ == "__main__":
    raise SystemExit("Use scripts/build_onedir.py to build the release candidate.")
