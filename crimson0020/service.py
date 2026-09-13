import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .archive import (
    ArchiveFormatError,
    extract_localization_palocs,
    parse_pamt,
    repack_localization_palocs,
    validate_localization_package,
)
from .catalog import CatalogBuild, build_catalog_replacements, palocs_to_catalog
from .checksums import pa_checksum
from .languages import LanguageSpec, get_language_spec
from .paloc import (
    json_data_to_paloc,
    paloc_to_json_data,
    read_paloc_json,
    write_paloc_json,
)


@dataclass(frozen=True)
class ExportResult:
    json_path: Path
    backup_json: Path | None
    entry_count: int
    paloc_size: int
    language: str = "English"
    paloc_count: int = 1


@dataclass(frozen=True)
class BuildResult:
    output_package: Path
    backup_package: Path | None
    entry_count: int
    paloc_size: int
    compressed_size: int
    paz_size: int
    language: str = "English"
    paloc_count: int = 1
    matched_entries: int = 0
    new_entries: int = 0
    removed_entries: int = 0
    changed_originals: int = 0

    @property
    def output_0020(self) -> Path:
        return self.output_package

    @property
    def backup_0020(self) -> Path | None:
        return self.backup_package


def normalize_localization_folder(
    selected_path: Path | str,
    spec: LanguageSpec,
) -> Path:
    selected = Path(selected_path).expanduser()
    if selected.name != spec.package_folder and (selected / spec.package_folder).is_dir():
        selected = selected / spec.package_folder
    selected = selected.resolve()
    if selected.name != spec.package_folder or not selected.is_dir():
        raise ArchiveFormatError(
            f"Please select a folder named {spec.package_folder} containing "
            "0.pamt and 0.paz."
        )
    if not (selected / "0.pamt").is_file() or not (selected / "0.paz").is_file():
        raise ArchiveFormatError(
            f"Folder {spec.package_folder} must contain both 0.pamt and 0.paz."
        )
    index = parse_pamt(selected / "0.pamt", selected)
    validate_localization_package(index, spec)
    return selected


def normalize_0020_folder(selected_path: Path | str) -> Path:
    return normalize_localization_folder(selected_path, get_language_spec("English"))


def export_localization_to_json(
    source_package: Path | str,
    output_json: Path | str,
    spec: LanguageSpec,
) -> ExportResult:
    source = normalize_localization_folder(source_package, spec)
    json_path = Path(output_json).expanduser().resolve()
    palocs = extract_localization_palocs(source, spec)
    if len(palocs) == 1 and palocs[0].path.lower() == spec.paloc_entry_path.lower():
        json_data = paloc_to_json_data(palocs[0].data)
    else:
        json_data = palocs_to_catalog(palocs, spec.package_folder)
    backup: Path | None = None
    if json_path.is_file():
        backup = _backup_json_path(json_path)
        os.replace(json_path, backup)
    try:
        entry_count = write_paloc_json(json_data, json_path)
    except Exception:
        if backup is not None and backup.exists() and not json_path.exists():
            os.replace(backup, json_path)
            backup = None
        raise
    return ExportResult(
        json_path=json_path,
        backup_json=backup,
        entry_count=entry_count,
        paloc_size=sum(len(paloc.data) for paloc in palocs),
        language=spec.name,
        paloc_count=len(palocs),
    )


def export_0020_to_json(
    source_0020: Path | str,
    output_json: Path | str,
) -> ExportResult:
    return export_localization_to_json(
        source_0020,
        output_json,
        get_language_spec("English"),
    )


def _backup_json_path(json_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = json_path.with_name(f"{json_path.name}.backup-{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = json_path.with_name(
            f"{json_path.name}.backup-{timestamp}-{suffix}"
        )
        suffix += 1
    return candidate


def _backup_path(output_parent: Path, package_folder: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = output_parent / f"{package_folder}.backup-{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_parent / f"{package_folder}.backup-{timestamp}-{suffix}"
        suffix += 1
    return candidate


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _remove_staging(
    staging: Path,
    output_parent: Path,
    package_folder: str,
) -> None:
    resolved_staging = staging.resolve()
    resolved_parent = output_parent.resolve()
    if (
        resolved_staging.parent != resolved_parent
        or not staging.name.startswith(f".{package_folder}-building-")
    ):
        raise RuntimeError("Refusing to remove an unexpected staging directory.")
    if staging.exists():
        shutil.rmtree(staging)


def _verify_built_package(
    package: Path,
    expected_palocs: tuple[tuple[str, bytes], ...],
    spec: LanguageSpec,
) -> None:
    rebuilt_palocs = extract_localization_palocs(package, spec)
    rebuilt_by_path = {paloc.path.lower(): paloc.data for paloc in rebuilt_palocs}
    expected_by_path = {path.lower(): data for path, data in expected_palocs}
    if rebuilt_by_path != expected_by_path:
        raise ArchiveFormatError(
            f"PALOC verification failed after rebuilding {spec.package_folder}."
        )
    index = parse_pamt(package / "0.pamt", package)
    entries = validate_localization_package(index, spec)
    paz_data = entries[0].paz_path.read_bytes()
    paz_record = index.paz_records[0]
    if len(paz_data) != paz_record.stored_size:
        raise ArchiveFormatError("PAZ size in 0.pamt does not match the rebuilt 0.paz.")
    if pa_checksum(paz_data) != paz_record.stored_checksum:
        raise ArchiveFormatError("PAZ checksum in 0.pamt does not match the rebuilt 0.paz.")
    pamt_data = (package / "0.pamt").read_bytes()
    if pa_checksum(pamt_data[12:]) != int.from_bytes(pamt_data[:4], "little"):
        raise ArchiveFormatError("0.pamt header checksum verification failed.")


def build_localization_from_json(
    source_package: Path | str,
    json_path: Path | str,
    output_parent: Path | str,
    spec: LanguageSpec,
) -> BuildResult:
    source = normalize_localization_folder(source_package, spec)
    source_json = Path(json_path).expanduser().resolve()
    if not source_json.is_file():
        raise ValueError(f"JSON file does not exist: {source_json}")
    source_palocs = extract_localization_palocs(source, spec)
    json_data = read_paloc_json(source_json)
    metadata = json_data.get("metadata", {})
    is_catalog = isinstance(metadata, dict) and "format_version" in metadata
    if (
        len(source_palocs) == 1
        and source_palocs[0].path.lower() == spec.paloc_entry_path.lower()
        and not is_catalog
    ):
        paloc_data = json_data_to_paloc(json_data)
        entry_count = len(json_data["entries"])
        catalog_build = CatalogBuild(
            replacements=((source_palocs[0].path, paloc_data),),
            entry_count=entry_count,
            paloc_count=1,
            matched_entries=entry_count,
            new_entries=0,
            removed_entries=0,
            changed_originals=0,
        )
    else:
        catalog_build = build_catalog_replacements(
            json_data,
            source_palocs,
            spec.package_folder,
        )

    destination_parent = Path(output_parent).expanduser().resolve()
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = destination_parent / spec.package_folder
    if destination == source or _is_within(destination_parent, source):
        raise ValueError(
            f"Output must be outside the source {spec.package_folder} folder."
        )

    staging = destination_parent / (
        f".{spec.package_folder}-building-{uuid.uuid4().hex}"
    )
    backup: Path | None = None
    try:
        shutil.copytree(source, staging)
        stats = repack_localization_palocs(
            staging,
            dict(catalog_build.replacements),
            spec,
        )
        _verify_built_package(staging, catalog_build.replacements, spec)

        if destination.exists():
            backup = _backup_path(destination_parent, spec.package_folder)
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
            _remove_staging(staging, destination_parent, spec.package_folder)

    return BuildResult(
        output_package=destination,
        backup_package=backup,
        entry_count=catalog_build.entry_count,
        paloc_size=stats.original_size,
        compressed_size=stats.compressed_size,
        paz_size=stats.paz_size,
        language=spec.name,
        paloc_count=catalog_build.paloc_count,
        matched_entries=catalog_build.matched_entries,
        new_entries=catalog_build.new_entries,
        removed_entries=catalog_build.removed_entries,
        changed_originals=catalog_build.changed_originals,
    )


def build_0020_from_json(
    source_0020: Path | str,
    json_path: Path | str,
    output_parent: Path | str,
) -> BuildResult:
    return build_localization_from_json(
        source_0020,
        json_path,
        output_parent,
        get_language_spec("English"),
    )
