import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .archive import (
    ArchiveFormatError,
    append_entries_to_last_paz,
    extract_archive_entry,
    parse_pamt,
)
from .checksums import pa_checksum, pa_checksum_file
from .languages import LanguageSpec


class FontFormatError(ValueError):
    """Raised when a selected font source or TTF file is invalid."""


@dataclass(frozen=True)
class FontExportResult:
    output_folder: Path
    backup_folder: Path | None
    font_count: int
    total_size: int


@dataclass(frozen=True)
class FontBuildResult:
    output_0012: Path
    backup_0012: Path | None
    font_count: int
    paz_filename: str
    paz_size: int
    pamt_checksum: int


def _is_ttf(data: bytes) -> bool:
    return data[:4] in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1")


def _validate_ttf(data: bytes, filename: str) -> None:
    if not _is_ttf(data):
        raise FontFormatError(
            f"{filename} is not a valid TrueType/OpenType font file."
        )


def normalize_font_source(selected_path: Path | str, spec: LanguageSpec) -> Path:
    selected = Path(selected_path).expanduser()
    if selected.name != "0012" and (selected / "0012").is_dir():
        selected = selected / "0012"
    selected = selected.resolve()
    if selected.name != "0012" or not selected.is_dir():
        raise FontFormatError(
            "Please select the 0012 folder containing the game's font archive."
        )
    if not (selected / "0.pamt").is_file():
        raise FontFormatError("Folder 0012 must contain 0.pamt.")
    index = parse_pamt(selected / "0.pamt", selected)
    for entry_path in spec.font_entry_paths:
        entry = index.find_entry(entry_path)
        if entry.compression_type != 2:
            raise FontFormatError(
                f"Font entry {entry_path} must use LZ4 compression type 2."
            )
        _validate_ttf(extract_archive_entry(selected, entry_path), Path(entry_path).name)
    return selected


def _timestamped_sibling(path: Path, label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.name}.{label}-{timestamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{label}-{timestamp}-{counter}")
        counter += 1
    return candidate


def _remove_staging(staging: Path, parent: Path, prefix: str) -> None:
    if staging.resolve().parent != parent.resolve() or not staging.name.startswith(prefix):
        raise RuntimeError("Refusing to remove an unexpected font staging directory.")
    if staging.exists():
        shutil.rmtree(staging)


def export_fonts(
    source_0012: Path | str,
    output_folder: Path | str,
    spec: LanguageSpec,
) -> FontExportResult:
    source = normalize_font_source(source_0012, spec)
    destination = Path(output_folder).expanduser().resolve()
    if (
        destination == source
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("Font output must be outside the source 0012 folder.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}-extracting-{uuid.uuid4().hex}"
    backup: Path | None = None
    total_size = 0
    try:
        staging.mkdir()
        for entry_path in spec.font_entry_paths:
            filename = Path(entry_path).name
            data = extract_archive_entry(source, entry_path)
            _validate_ttf(data, filename)
            (staging / filename).write_bytes(data)
            total_size += len(data)
        if destination.exists():
            backup = _timestamped_sibling(destination, "backup")
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except Exception:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
                backup = None
            raise
    finally:
        if staging.exists():
            _remove_staging(staging, destination.parent, f".{destination.name}-extracting-")
    return FontExportResult(destination, backup, 3, total_size)


def _load_font_inputs(font_folder: Path, spec: LanguageSpec) -> dict[str, bytes]:
    replacements: dict[str, bytes] = {}
    for entry_path in spec.font_entry_paths:
        filename = Path(entry_path).name
        matches = [path for path in font_folder.rglob(filename) if path.is_file()]
        if not matches:
            raise FontFormatError(f"Required font file was not found: {filename}")
        if len(matches) > 1:
            raise FontFormatError(f"Multiple font files named {filename} were found.")
        data = matches[0].read_bytes()
        _validate_ttf(data, filename)
        replacements[entry_path] = data
    return replacements


def _verify_font_package(
    package: Path,
    replacements: dict[str, bytes],
    paz_filename: str,
) -> int:
    index = parse_pamt(package / "0.pamt", package)
    record_index = int(Path(paz_filename).stem) - int(index.pamt_path.stem)
    if not 0 <= record_index < len(index.paz_records):
        raise ArchiveFormatError("Built font PAZ index is outside the PAMT table.")
    for entry_path, expected in replacements.items():
        if extract_archive_entry(package, entry_path) != expected:
            raise ArchiveFormatError(f"Font verification failed for {entry_path}.")
    paz_path = package / paz_filename
    record = index.paz_records[record_index]
    if record.stored_size != paz_path.stat().st_size:
        raise ArchiveFormatError("Font PAZ size does not match 0.pamt.")
    if record.stored_checksum != pa_checksum_file(paz_path):
        raise ArchiveFormatError("Font PAZ checksum does not match 0.pamt.")
    pamt_data = (package / "0.pamt").read_bytes()
    pamt_checksum = pa_checksum(pamt_data[12:])
    if pamt_checksum != int.from_bytes(pamt_data[:4], "little"):
        raise ArchiveFormatError("Font 0.pamt header checksum verification failed.")
    return pamt_checksum


def build_font_package(
    source_0012: Path | str,
    font_folder: Path | str,
    output_parent: Path | str,
    spec: LanguageSpec,
) -> FontBuildResult:
    source = normalize_font_source(source_0012, spec)
    selected_fonts = Path(font_folder).expanduser().resolve()
    if not selected_fonts.is_dir():
        raise FontFormatError(f"Font folder does not exist: {selected_fonts}")
    replacements = _load_font_inputs(selected_fonts, spec)

    destination_parent = Path(output_parent).expanduser().resolve()
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = destination_parent / "0012"
    if destination == source or destination_parent == source or source in destination_parent.parents:
        raise ValueError("Font output must be outside the source 0012 folder.")

    source_index = parse_pamt(source / "0.pamt", source)
    last_index = len(source_index.paz_records) - 1
    last_paz_name = f"{int(source_index.pamt_path.stem) + last_index}.paz"
    source_last_paz = source / last_paz_name
    if not source_last_paz.is_file():
        raise FontFormatError(f"Source 0012 is missing {last_paz_name}.")

    staging = destination_parent / f".0012-font-building-{uuid.uuid4().hex}"
    backup: Path | None = None
    try:
        staging.mkdir()
        shutil.copy2(source / "0.pamt", staging / "0.pamt")
        shutil.copy2(source_last_paz, staging / last_paz_name)
        stats = append_entries_to_last_paz(staging, replacements)
        pamt_checksum = _verify_font_package(staging, replacements, stats.paz_filename)
        if destination.exists():
            backup = _timestamped_sibling(destination, "backup")
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except Exception:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
                backup = None
            raise
    finally:
        if staging.exists():
            _remove_staging(staging, destination_parent, ".0012-font-building-")
    return FontBuildResult(
        output_0012=destination,
        backup_0012=backup,
        font_count=len(replacements),
        paz_filename=stats.paz_filename,
        paz_size=stats.paz_size,
        pamt_checksum=pamt_checksum,
    )
