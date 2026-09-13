import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


APP_FOLDER_NAME = "CrimsonDesertLocalizationBuilder"
ARCHIVE_NAME = f"{APP_FOLDER_NAME}-portable.zip"
OFFICIAL_RUNTIME_FILENAME = "python-3.14.7-amd64.zip"
OFFICIAL_RUNTIME_URL = (
    "https://www.python.org/ftp/python/3.14.7/python-3.14.7-amd64.zip"
)
OFFICIAL_RUNTIME_SHA256 = (
    "ac1a727a71738e11de80b76e975f9b8a258aea6412bfc31696b929d59c6aafd0"
)
PORTABLE_REQUIREMENTS_FILENAME = "requirements-portable.txt"
FIXED_ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)
APPLICATION_ASSET_FILENAMES = (
    "crimson-desert-localization.ico",
    "crimson-desert-localization.png",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime_archive(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Official Python runtime archive not found: {path}")
    actual = sha256_file(path)
    if actual.casefold() != OFFICIAL_RUNTIME_SHA256.casefold():
        raise ValueError(
            "Official Python runtime SHA-256 mismatch: "
            f"expected {OFFICIAL_RUNTIME_SHA256}, found {actual}."
        )


def download_runtime_archive(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_runtime_archive(destination)
        return destination
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with urllib.request.urlopen(OFFICIAL_RUNTIME_URL, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        verify_runtime_archive(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            extracted = (root / member.filename).resolve()
            if not extracted.is_relative_to(root):
                raise ValueError(f"ZIP contains an unsafe path: {member.filename}")
        source.extractall(destination)


def copy_application_sources(project: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in (
        "app.py",
        "sitecustomize.py",
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
    ):
        source = project / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required application source is missing: {source}")
        shutil.copy2(source, destination / filename)
    package_source = project / "crimson0020"
    for source in sorted(package_source.rglob("*.py")):
        relative = source.relative_to(project)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    asset_source = package_source / "assets"
    for filename in APPLICATION_ASSET_FILENAMES:
        source = asset_source / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required application asset is missing: {source}")
        relative = source.relative_to(project)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in sorted((package_source / "webui").rglob("*")):
        if source.is_file():
            relative = source.relative_to(project)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for compatibility_package in ("proxy_tools", "cloudscraper"):
        package_root = project / compatibility_package
        if not package_root.is_dir():
            raise FileNotFoundError(
                f"Required compatibility source is missing: {package_root}"
            )
        for source in sorted(package_root.rglob("*.py")):
            relative = source.relative_to(project)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def write_isolated_path_file(runtime: Path) -> Path:
    path_file = runtime / "python314._pth"
    path_file.write_text(
        ".\nDLLs\nLib\nLib\\site-packages\n..\\app\n",
        encoding="utf-8",
        newline="\n",
    )
    return path_file


def write_launcher(candidate: Path) -> Path:
    launcher = candidate / "START_TOOL.cmd"
    launcher.write_text(
        "@echo off\n"
        "setlocal\n"
        "cd /d \"%~dp0\"\n"
        "start \"\" \"%~dp0runtime\\pythonw.exe\" -E -s -X utf8 "
        "-B \"%~dp0app\\app.py\"\n",
        encoding="utf-8",
        newline="\r\n",
    )
    return launcher


def _write_build_info(candidate: Path, requirements: tuple[str, ...]) -> None:
    build_info = {
        "artifact_type": "portable-source-with-official-cpython-runtime",
        "application_folder": APP_FOLDER_NAME,
        "custom_executable_packer": None,
        "launcher": "START_TOOL.cmd",
        "python_runtime": {
            "filename": OFFICIAL_RUNTIME_FILENAME,
            "sha256": OFFICIAL_RUNTIME_SHA256,
            "source": OFFICIAL_RUNTIME_URL,
        },
        "requirements": requirements,
    }
    (candidate / "BUILD_INFO.json").write_text(
        json.dumps(build_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_checksums(candidate: Path, checksum_path: Path | None = None) -> Path:
    checksum_path = checksum_path or candidate / "SHA256SUMS.txt"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_checksum_path = checksum_path.resolve()
    files = sorted(
        path
        for path in candidate.rglob("*")
        if path.is_file() and path.resolve() != resolved_checksum_path
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(candidate).as_posix()}"
        for path in files
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return checksum_path


def create_deterministic_zip(candidate: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as output:
        for source in sorted(path for path in candidate.rglob("*") if path.is_file()):
            archive_name = (Path(candidate.name) / source.relative_to(candidate)).as_posix()
            info = zipfile.ZipInfo(archive_name, FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, source.read_bytes(), compresslevel=9)


def read_runtime_requirements(project: Path) -> tuple[str, ...]:
    lines = (project / PORTABLE_REQUIREMENTS_FILENAME).read_text(
        encoding="utf-8"
    ).splitlines()
    return tuple(
        line.strip().removesuffix("\\").strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith(("#", "--hash="))
    )


def install_dependencies(runtime: Path, project: Path, staging: Path) -> None:
    third_party = staging / "site-packages"
    command = (
        str(runtime / "python.exe"),
        "-E",
        "-s",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-compile",
        "--no-deps",
        "--require-hashes",
        "--target",
        str(third_party),
        "-r",
        str(project / PORTABLE_REQUIREMENTS_FILENAME),
    )
    subprocess.run(command, check=True)
    bundled_packages = runtime / "Lib" / "site-packages"
    shutil.rmtree(bundled_packages)
    third_party.replace(bundled_packages)


def prune_runtime(runtime: Path) -> None:
    for relative in (
        Path("Doc"),
        Path("include"),
        Path("Lib") / "site-packages" / "bin",
        Path("Lib") / "venv",
        Path("Lib") / "ensurepip",
        Path("Lib") / "idlelib",
        Path("Lib") / "test",
    ):
        shutil.rmtree(runtime / relative, ignore_errors=True)
    site_packages = runtime / "Lib" / "site-packages"
    for record in site_packages.glob("*.dist-info/RECORD"):
        rows = list(csv.reader(record.read_text(encoding="utf-8").splitlines()))
        retained = [row for row in rows if row and not row[0].startswith("../")]
        output = io.StringIO(newline="")
        csv.writer(output, lineterminator="\n").writerows(retained)
        record.write_text(output.getvalue(), encoding="utf-8", newline="\n")


def remove_generated_bytecode(root: Path) -> None:
    caches = sorted(
        (path for path in root.rglob("__pycache__") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for cache in caches:
        shutil.rmtree(cache)
    for pattern in ("*.pyc", "*.pyo"):
        for compiled in root.rglob(pattern):
            compiled.unlink()


def build_native_helper(project: Path, application: Path) -> None:
    output = application / "crimson0020" / "native" / "pa_checksum.dll"
    subprocess.run(
        (
            sys.executable,
            str(project / "build_native.py"),
            "--output",
            str(output),
        ),
        check=True,
    )
    for suffix in (".lib", ".pdb"):
        output.with_suffix(suffix).unlink(missing_ok=True)


def _smoke_test(candidate: Path) -> None:
    script = (
        "import clr, cryptography, lz4, webview; "
        "from crimson0020.webview_app import validate_web_assets; "
        "from crimson0020.native_checksum import native_checksum_available; "
        "validate_web_assets(); "
        "assert native_checksum_available(), 'native checksum library unavailable'"
    )
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run(
        (
            str(candidate / "runtime" / "python.exe"),
            "-E",
            "-s",
            "-X",
            "utf8",
            "-B",
            "-c",
            script,
        ),
        cwd=candidate,
        check=True,
        timeout=30,
        creationflags=creation_flags,
    )


def build_portable(
    project: Path,
    output_root: Path,
    runtime_archive: Path | None = None,
) -> tuple[Path, Path]:
    if os.name != "nt":
        raise OSError("The portable Windows candidate must be built on Windows.")
    project = project.resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidate = output_root / APP_FOLDER_NAME
    release_zip = output_root / ARCHIVE_NAME
    if candidate.exists() or release_zip.exists():
        raise FileExistsError(
            f"Candidate output already exists; use a new output directory: {output_root}"
        )
    archive = (
        runtime_archive.expanduser().resolve()
        if runtime_archive
        else output_root / "downloads" / OFFICIAL_RUNTIME_FILENAME
    )
    if runtime_archive:
        verify_runtime_archive(archive)
    else:
        archive = download_runtime_archive(archive)

    with tempfile.TemporaryDirectory(prefix="portable-build-", dir=output_root) as temp:
        staging = Path(temp)
        staged_candidate = staging / APP_FOLDER_NAME
        application = staged_candidate / "app"
        runtime = staged_candidate / "runtime"
        copy_application_sources(project, application)
        build_native_helper(project, application)
        safe_extract_zip(archive, runtime)
        install_dependencies(runtime, project, staging)
        shutil.rmtree(runtime / "libs", ignore_errors=True)
        prune_runtime(runtime)
        write_isolated_path_file(runtime)
        write_launcher(staged_candidate)
        _write_build_info(staged_candidate, read_runtime_requirements(project))
        _smoke_test(staged_candidate)
        remove_generated_bytecode(staged_candidate)
        write_checksums(staged_candidate)
        staged_candidate.replace(candidate)

    create_deterministic_zip(candidate, release_zip)
    return candidate, release_zip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a transparent Windows folder using the official CPython runtime."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parent.parent
    candidate, release_zip = build_portable(
        project,
        args.output_root,
        args.runtime_archive,
    )
    print(f"Portable candidate: {candidate}")
    print(f"Deterministic archive: {release_zip}")
    print(f"Archive SHA-256: {sha256_file(release_zip)}")


if __name__ == "__main__":
    main()
