import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_launcher import compile_launcher  # noqa: E402
from scripts.build_portable import (  # noqa: E402
    OFFICIAL_RUNTIME_FILENAME,
    OFFICIAL_RUNTIME_SHA256,
    OFFICIAL_RUNTIME_URL,
    build_native_helper,
    copy_application_sources,
    create_deterministic_zip,
    download_runtime_archive,
    install_dependencies,
    prune_runtime,
    read_runtime_requirements,
    remove_generated_bytecode,
    safe_extract_zip,
    sha256_file,
    verify_runtime_archive,
    write_checksums,
    write_isolated_path_file,
)


APP_FOLDER_NAME = "CrimsonDesertLocalizationBuilder"
INTERNAL_FOLDER_NAME = "_internal"
ARCHIVE_NAME = f"{APP_FOLDER_NAME}-windows-x64.zip"
LAUNCHER_EXE_NAME = "Crimson.Desert.Localization.Builder.exe"
APPLICATION_ICON_RELATIVE = Path(
    "crimson0020/assets/crimson-desert-localization.ico"
)
APPLICATION_ICON_PREVIEW_RELATIVE = Path(
    "crimson0020/assets/crimson-desert-localization.png"
)
OFFICIAL_PYTHONW_SHA256 = (
    "c197268f7e7cf2848b8c1ae59bbd0e0c14defe668a2d365302137ea929b47769"
)
REQUIRED_RUNTIME_FILES = (
    "pythonw.exe",
    "python3.dll",
    "python314.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
)


def prepare_internal_runtime(
    runtime: Path,
    expected_pythonw_sha256: str | None = OFFICIAL_PYTHONW_SHA256,
) -> Path:
    pythonw = runtime / "pythonw.exe"
    if not pythonw.is_file():
        raise FileNotFoundError(f"Official Python GUI executable is missing: {pythonw}")
    if expected_pythonw_sha256 is not None:
        actual = sha256_file(pythonw)
        if actual.casefold() != expected_pythonw_sha256.casefold():
            raise ValueError(
                "Official pythonw.exe SHA-256 mismatch: "
                f"expected {expected_pythonw_sha256}, found {actual}."
            )

    for filename in REQUIRED_RUNTIME_FILES:
        dependency = runtime / filename
        if not dependency.is_file():
            raise FileNotFoundError(
                f"Official Python runtime dependency is missing: {dependency}"
            )
    (runtime / "python.exe").unlink(missing_ok=True)
    return pythonw


def validate_candidate_layout(candidate: Path) -> None:
    root_entries = {path.name for path in candidate.iterdir()}
    expected_root_entries = {LAUNCHER_EXE_NAME, INTERNAL_FOLDER_NAME}
    if root_entries != expected_root_entries:
        raise ValueError(
            "The candidate root must contain exactly the launcher EXE and "
            f"{INTERNAL_FOLDER_NAME}; found: {sorted(root_entries)}"
        )
    required_files = (
        Path(LAUNCHER_EXE_NAME),
        *(
            Path(INTERNAL_FOLDER_NAME) / "runtime" / filename
            for filename in REQUIRED_RUNTIME_FILES
        ),
        Path(INTERNAL_FOLDER_NAME) / "runtime" / "python314._pth",
        Path(INTERNAL_FOLDER_NAME) / "app" / "app.py",
        Path(INTERNAL_FOLDER_NAME)
        / "app"
        / "crimson0020"
        / "native"
        / "pa_checksum.dll",
        Path(INTERNAL_FOLDER_NAME) / "app" / APPLICATION_ICON_RELATIVE,
        Path(INTERNAL_FOLDER_NAME) / "app" / APPLICATION_ICON_PREVIEW_RELATIVE,
        Path(INTERNAL_FOLDER_NAME) / "BUILD_INFO.json",
        Path(INTERNAL_FOLDER_NAME) / "SHA256SUMS.txt",
    )
    for relative in required_files:
        path = candidate / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required one-folder file is missing: {relative}")
    if (candidate / INTERNAL_FOLDER_NAME / "runtime" / "python.exe").exists():
        raise ValueError("The build-only python.exe must not remain in the candidate.")


def write_build_info(internal: Path, requirements: tuple[str, ...]) -> Path:
    build_info = {
        "artifact_type": "one-folder-microsoft-dotnet-launcher",
        "application_folder": APP_FOLDER_NAME,
        "custom_executable_packer": None,
        "embedded_payload": False,
        "first_run_extraction": False,
        "internal_folder": INTERNAL_FOLDER_NAME,
        "layout": "pre-extracted-one-folder",
        "launcher": LAUNCHER_EXE_NAME,
        "application_icon": APPLICATION_ICON_RELATIVE.as_posix(),
        "launcher_icon_embedded": True,
        "launcher_source": {
            "compiler": "Microsoft Roslyn C# compiler",
            "framework_target": ".NET Framework 4.x",
            "source": "launcher/ToolLauncher.cs",
        },
        "python_runtime": {
            "filename": OFFICIAL_RUNTIME_FILENAME,
            "pythonw_sha256": OFFICIAL_PYTHONW_SHA256,
            "sha256": OFFICIAL_RUNTIME_SHA256,
            "source": OFFICIAL_RUNTIME_URL,
        },
        "requirements": requirements,
    }
    destination = internal / "BUILD_INFO.json"
    destination.write_text(
        json.dumps(build_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def smoke_test_launcher(candidate: Path) -> None:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run(
        (str(candidate / LAUNCHER_EXE_NAME), "--smoke-test"),
        cwd=candidate,
        check=True,
        timeout=30,
        creationflags=creation_flags,
    )


def build_onedir(
    project: Path,
    output_root: Path,
    runtime_archive: Path | None = None,
) -> tuple[Path, Path, Path]:
    if os.name != "nt":
        raise OSError("The one-folder Windows candidate must be built on Windows.")
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

    with tempfile.TemporaryDirectory(prefix="onedir-build-", dir=output_root) as temp:
        staging = Path(temp)
        staged_candidate = staging / APP_FOLDER_NAME
        internal = staged_candidate / INTERNAL_FOLDER_NAME
        application = internal / "app"
        runtime = internal / "runtime"

        copy_application_sources(project, application)
        build_native_helper(project, application)
        safe_extract_zip(archive, runtime)
        install_dependencies(runtime, project, staging)
        shutil.rmtree(runtime / "libs", ignore_errors=True)
        prune_runtime(runtime)
        prepare_internal_runtime(runtime)
        write_isolated_path_file(runtime)
        compile_launcher(
            project / "launcher" / "ToolLauncher.cs",
            staged_candidate / LAUNCHER_EXE_NAME,
            project / APPLICATION_ICON_RELATIVE,
        )
        write_build_info(internal, read_runtime_requirements(project))
        remove_generated_bytecode(staged_candidate)
        write_checksums(staged_candidate, internal / "SHA256SUMS.txt")
        validate_candidate_layout(staged_candidate)
        smoke_test_launcher(staged_candidate)
        staged_candidate.replace(candidate)

    create_deterministic_zip(candidate, release_zip)
    return candidate, candidate / LAUNCHER_EXE_NAME, release_zip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the transparent Windows one-folder release."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path)
    args = parser.parse_args(argv)
    candidate, launcher, release_zip = build_onedir(
        PROJECT_ROOT,
        args.output_root,
        args.runtime_archive,
    )
    print(f"One-folder candidate: {candidate}")
    print(f"Main executable: {launcher}")
    print(f"Executable SHA-256: {sha256_file(launcher)}")
    print(f"Deterministic archive: {release_zip}")
    print(f"Archive SHA-256: {sha256_file(release_zip)}")


if __name__ == "__main__":
    main()
