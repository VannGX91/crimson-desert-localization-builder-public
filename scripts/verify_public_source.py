import argparse
import re
import subprocess
from pathlib import Path


MAX_PUBLIC_FILE_BYTES = 2_000_000
FORBIDDEN_DIRECTORY_NAMES = {
    ".venv",
    "__pycache__",
    "build",
    "output_repack",
    "output_test",
    "output_unpack",
    "release",
}
FORBIDDEN_FILE_NAMES = {
    ".coverage",
    ".env",
    "config.json",
    "custom_openai_config.json",
}
FORBIDDEN_GENERATED_PATTERNS = (
    "translation-knowledge-v1*.sqlite3*",
    "*.autotranslate-checkpoint.jsonl",
    "*.autotranslate-failures.jsonl",
)
FORBIDDEN_SUFFIXES = {
    ".dll",
    ".exe",
    ".lib",
    ".otf",
    ".pamt",
    ".papgt",
    ".paz",
    ".pdb",
    ".pyc",
    ".ttf",
}
PERSONAL_PATH_PATTERNS = (
    re.compile(r"(?i)[a-z]:[\\/]users[\\/][^\\/\s]+(?:[\\/]|$)"),
    re.compile(r"(?i)/(?:home|users)/[^/\s]+/"),
)
CREDENTIAL_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgh[oprsu]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
)
PUBLIC_BINARY_ASSETS = frozenset(
    {
        "crimson0020/assets/crimson-desert-localization.ico",
        "crimson0020/assets/crimson-desert-localization.png",
    }
)


def _repository_files(root: Path) -> tuple[Path, ...]:
    if (root / ".git").exists():
        result = subprocess.run(
            ("git", "-C", str(root), "ls-files", "-z"),
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            relative_paths = result.stdout.decode("utf-8").split("\0")
            return tuple(root / path for path in relative_paths if path)
    return tuple(
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )


def _path_violations(relative_path: Path) -> tuple[str, ...]:
    display = relative_path.as_posix()
    violations: list[str] = []
    forbidden_parts = FORBIDDEN_DIRECTORY_NAMES.intersection(relative_path.parts)
    if forbidden_parts:
        violations.append(
            f"{display}: generated/private directory {sorted(forbidden_parts)[0]}"
        )
    if relative_path.name.casefold() in FORBIDDEN_FILE_NAMES:
        violations.append(f"{display}: private/generated file name")
    if any(relative_path.match(pattern) for pattern in FORBIDDEN_GENERATED_PATTERNS):
        violations.append(f"{display}: private/generated runtime data")
    if relative_path.suffix.casefold() in FORBIDDEN_SUFFIXES:
        violations.append(f"{display}: forbidden binary/game suffix {relative_path.suffix}")
    return tuple(violations)


def _content_violations(path: Path, relative_path: Path) -> tuple[str, ...]:
    display = relative_path.as_posix()
    file_size = path.stat().st_size
    if file_size > MAX_PUBLIC_FILE_BYTES:
        return (f"{display}: unexpected large file ({file_size} bytes)",)
    if display in PUBLIC_BINARY_ASSETS:
        return ()
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return (f"{display}: non-UTF-8 or binary content",)
    violations: list[str] = []
    if any(pattern.search(content) for pattern in PERSONAL_PATH_PATTERNS):
        violations.append(f"{display}: contains a personal path")
    if any(pattern.search(content) for pattern in CREDENTIAL_PATTERNS):
        violations.append(f"{display}: contains credential-like content")
    return tuple(violations)


def collect_violations(source_root: str | Path) -> tuple[str, ...]:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Source root does not exist: {root}")
    violations: list[str] = []
    for path in _repository_files(root):
        relative_path = path.relative_to(root)
        violations.extend(_path_violations(relative_path))
        violations.extend(_content_violations(path, relative_path))
    return tuple(sorted(set(violations)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reject private paths, credentials, game data, and build artifacts."
    )
    parser.add_argument("source_root", nargs="?", default=".")
    args = parser.parse_args()
    violations = collect_violations(args.source_root)
    if violations:
        print("Public source verification failed:")
        for violation in violations:
            print(f"- {violation}")
        raise SystemExit(1)
    print("Public source verification passed.")


if __name__ == "__main__":
    main()
