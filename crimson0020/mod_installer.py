import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .archive import (
    extract_localization_palocs,
    parse_pamt,
    validate_localization_package,
)
from .checksums import pa_checksum, pa_checksum_file
from .font_service import normalize_font_source
from .installer import (
    BACKUP_FOLDER_NAME,
    BackupFormatError,
    BackupNotFoundError,
    GamePathError,
    InstallationError,
    UninstallationError,
    uninstall_0020,
)
from .languages import LanguageSpec
from .metadata import MetadataFormatError, parse_papgt, patch_folder_crc
from .service import normalize_localization_folder


MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 2


@dataclass(frozen=True)
class PackageInstallPlan:
    folder_name: str
    source_folder: Path
    filenames: tuple[str, ...]
    pamt_crc: int
    metadata_crc: int


@dataclass(frozen=True)
class ModInstallResult:
    game_root: Path
    metadata_path: Path
    backup_directory: Path
    installed_folders: tuple[str, ...]
    folder_crcs: tuple[tuple[str, int], ...]
    metadata_checksum: int


@dataclass(frozen=True)
class ModUninstallResult:
    game_root: Path
    metadata_path: Path
    restored_backup_directory: Path
    safety_backup_directory: Path
    restored_folders: tuple[str, ...]
    folder_crcs: tuple[tuple[str, int], ...]
    metadata_checksum: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pamt_crc(package: Path) -> int:
    pamt_path = package / "0.pamt"
    data = pamt_path.read_bytes()
    if len(data) < 12:
        raise MetadataFormatError(f"Source {package.name}\\0.pamt is too small.")
    stored = int.from_bytes(data[:4], "little")
    computed = pa_checksum(data[12:])
    if stored != computed:
        raise MetadataFormatError(
            f"Source {package.name}\\0.pamt header checksum does not match."
        )
    return computed


def _verify_paz_records(package: Path, filenames: tuple[str, ...]) -> None:
    index = parse_pamt(package / "0.pamt", package)
    base_index = int(index.pamt_path.stem)
    for filename in filenames:
        if filename == "0.pamt":
            continue
        paz_path = package / filename
        if not paz_path.is_file():
            raise MetadataFormatError(
                f"Source {package.name} is missing {filename}."
            )
        record_index = int(paz_path.stem) - base_index
        if not 0 <= record_index < len(index.paz_records):
            raise MetadataFormatError(
                f"Source {filename} is outside the 0.pamt PAZ table."
            )
        record = index.paz_records[record_index]
        if record.stored_size != paz_path.stat().st_size:
            raise MetadataFormatError(
                f"Source {filename} size does not match 0.pamt."
            )
        if record.stored_checksum != pa_checksum_file(paz_path):
            raise MetadataFormatError(
                f"Source {filename} checksum does not match 0.pamt."
            )


def _localization_plan(mod_root: Path, spec: LanguageSpec) -> PackageInstallPlan:
    package = normalize_localization_folder(mod_root, spec)
    index = parse_pamt(package / "0.pamt", package)
    validate_localization_package(index, spec)
    extract_localization_palocs(package, spec)
    filenames = ("0.pamt", "0.paz")
    _verify_paz_records(package, filenames)
    checksum = _pamt_crc(package)
    return PackageInstallPlan(
        spec.package_folder,
        package,
        filenames,
        checksum,
        checksum,
    )


def _font_plan(mod_root: Path, spec: LanguageSpec) -> PackageInstallPlan | None:
    font_package = mod_root / "0012"
    if not font_package.is_dir():
        return None
    package = normalize_font_source(font_package, spec)
    index = parse_pamt(package / "0.pamt", package)
    paz_names = {
        index.find_entry(entry_path).paz_path.name
        for entry_path in spec.font_entry_paths
    }
    filenames = ("0.pamt", *sorted(paz_names, key=lambda name: int(Path(name).stem)))
    _verify_paz_records(package, filenames)
    checksum = _pamt_crc(package)
    return PackageInstallPlan("0012", package, filenames, checksum, checksum)


def _normalize_mod_root(selected_path: Path | str, spec: LanguageSpec) -> Path:
    selected = Path(selected_path).expanduser().resolve()
    if selected.name == spec.package_folder:
        return selected.parent
    if not selected.is_dir():
        raise ValueError(f"Mod folder does not exist: {selected}")
    return selected


def _build_install_plan(
    selected_path: Path | str,
    spec: LanguageSpec,
) -> tuple[PackageInstallPlan, ...]:
    mod_root = _normalize_mod_root(selected_path, spec)
    localization = _localization_plan(mod_root, spec)
    font = _font_plan(mod_root, spec)
    return (localization, font) if font is not None else (localization,)


def _normalize_game_root(
    selected_path: Path | str,
    plan: tuple[PackageInstallPlan, ...] | None = None,
) -> Path:
    value = str(selected_path).strip()
    if not value:
        raise GamePathError("Please select the Crimson Desert game folder.")
    game_root = Path(value).expanduser().resolve()
    metadata_path = game_root / "meta" / "0.papgt"
    if not game_root.is_dir() or not metadata_path.is_file():
        raise GamePathError("The game folder must contain meta\\0.papgt.")
    if plan is not None:
        metadata = parse_papgt(metadata_path.read_bytes())
        for package in plan:
            game_package = game_root / package.folder_name
            if not game_package.is_dir():
                raise GamePathError(
                    f"The game folder must contain {package.folder_name}."
                )
            metadata.entry_for(package.folder_name)
            for filename in package.filenames:
                if not (game_package / filename).is_file():
                    raise GamePathError(
                        f"Game {package.folder_name} is missing {filename}."
                    )
    return game_root


def _unique_backup_directory(backup_root: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime(f"{prefix}-%Y%m%d-%H%M%S-%f")
    candidate = backup_root / timestamp
    counter = 1
    while candidate.exists():
        candidate = backup_root / f"{timestamp}-{counter}"
        counter += 1
    return candidate


def _write_manifest(
    path: Path,
    kind: str,
    plan: tuple[PackageInstallPlan, ...],
) -> None:
    backup_root = path.parent
    metadata_index = parse_papgt(
        (backup_root / "meta" / "0.papgt").read_bytes()
    )
    folders = []
    for package in plan:
        backup_package = backup_root / package.folder_name
        folders.append(
            {
                "name": package.folder_name,
                "pamt_crc": _pamt_crc(backup_package),
                "metadata_crc": metadata_index.entry_for(package.folder_name).crc,
                "files": [
                    {
                        "name": filename,
                        "sha256": _sha256(backup_package / filename),
                    }
                    for filename in package.filenames
                ],
            }
        )
    manifest = {
        "version": MANIFEST_VERSION,
        "kind": kind,
        "folders": folders,
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _remove_backup_staging(staging: Path, backup_root: Path, prefix: str) -> None:
    if (
        staging.resolve().parent != backup_root.resolve()
        or not staging.name.startswith(f".{prefix}-")
        or not staging.name.endswith(".tmp")
    ):
        raise RuntimeError("Refusing to remove an unexpected backup staging path.")
    if staging.exists():
        shutil.rmtree(staging)


def _create_backup(
    game_root: Path,
    prefix: str,
    plan: tuple[PackageInstallPlan, ...],
) -> Path:
    backup_root = game_root / "meta" / BACKUP_FOLDER_NAME
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = _unique_backup_directory(backup_root, prefix)
    staging = backup_root / f".{destination.name}-{uuid.uuid4().hex}.tmp"
    try:
        staging.mkdir()
        for package in plan:
            target = staging / package.folder_name
            target.mkdir()
            for filename in package.filenames:
                shutil.copy2(
                    game_root / package.folder_name / filename,
                    target / filename,
                )
        (staging / "meta").mkdir()
        shutil.copy2(
            game_root / "meta" / "0.papgt",
            staging / "meta" / "0.papgt",
        )
        _write_manifest(staging / MANIFEST_FILENAME, prefix, plan)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            _remove_backup_staging(staging, backup_root, prefix)
    return destination


def _replace_with_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_metadata_atomically(metadata_path: Path, data: bytes) -> None:
    temporary = metadata_path.with_name(
        f".{metadata_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_bytes(data)
        os.replace(temporary, metadata_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest_plan(backup: Path) -> tuple[PackageInstallPlan, ...]:
    manifest_path = backup / MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BackupFormatError(f"Cannot read backup manifest: {backup}") from error
    if manifest.get("version") != MANIFEST_VERSION:
        raise BackupFormatError(f"Unsupported backup manifest version: {backup}")
    folders = manifest.get("folders")
    if not isinstance(folders, list) or not folders:
        raise BackupFormatError(f"Backup manifest has no folders: {backup}")

    plan: list[PackageInstallPlan] = []
    for folder in folders:
        if not isinstance(folder, dict):
            raise BackupFormatError(f"Backup folder record is invalid: {backup}")
        folder_name = folder.get("name")
        if not isinstance(folder_name, str) or not re.fullmatch(r"\d{4}", folder_name):
            raise BackupFormatError(f"Backup folder name is unsafe: {backup}")
        file_records = folder.get("files")
        if not isinstance(file_records, list) or not file_records:
            raise BackupFormatError(f"Backup file list is invalid: {backup}")
        filenames: list[str] = []
        for record in file_records:
            if not isinstance(record, dict):
                raise BackupFormatError(f"Backup file record is invalid: {backup}")
            filename = record.get("name")
            expected_hash = record.get("sha256")
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(expected_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            ):
                raise BackupFormatError(f"Backup file record is unsafe: {backup}")
            source = backup / folder_name / filename
            if not source.is_file() or _sha256(source) != expected_hash:
                raise BackupFormatError(f"Backup file is missing or damaged: {source}")
            filenames.append(filename)
        if "0.pamt" not in filenames:
            raise BackupFormatError(f"Backup {folder_name} is missing 0.pamt.")
        stored_crc = folder.get("pamt_crc")
        computed_crc = _pamt_crc(backup / folder_name)
        if stored_crc != computed_crc:
            raise BackupFormatError(f"Backup {folder_name} CRC is inconsistent.")
        metadata_crc = folder.get("metadata_crc")
        if not isinstance(metadata_crc, int) or not 0 <= metadata_crc <= 0xFFFFFFFF:
            raise BackupFormatError(
                f"Backup {folder_name} metadata CRC is invalid."
            )
        plan.append(
            PackageInstallPlan(
                folder_name,
                backup / folder_name,
                tuple(filenames),
                computed_crc,
                metadata_crc,
            )
        )
    metadata = backup / "meta" / "0.papgt"
    if not metadata.is_file():
        raise BackupFormatError(f"Backup is missing meta\\0.papgt: {backup}")
    parse_papgt(metadata.read_bytes())
    return tuple(plan)


def _assert_game_matches_backup(
    game_root: Path,
    backup: Path,
    plan: tuple[PackageInstallPlan, ...],
) -> None:
    backup_plan = _manifest_plan(backup)
    if tuple((item.folder_name, item.filenames) for item in backup_plan) != tuple(
        (item.folder_name, item.filenames) for item in plan
    ):
        raise BackupFormatError("Backup plan changed unexpectedly.")
    for package in backup_plan:
        for filename in package.filenames:
            if _sha256(game_root / package.folder_name / filename) != _sha256(
                package.source_folder / filename
            ):
                raise MetadataFormatError(
                    f"Game {package.folder_name}\\{filename} changed while the operation was starting."
                )


def _restore_backup(backup: Path, game_root: Path) -> tuple[PackageInstallPlan, ...]:
    plan = _manifest_plan(backup)
    for package in plan:
        for filename in package.filenames:
            _replace_with_copy(
                package.source_folder / filename,
                game_root / package.folder_name / filename,
            )
    _replace_with_copy(
        backup / "meta" / "0.papgt",
        game_root / "meta" / "0.papgt",
    )
    for package in plan:
        for filename in package.filenames:
            if _sha256(package.source_folder / filename) != _sha256(
                game_root / package.folder_name / filename
            ):
                raise OSError(
                    f"Rollback verification failed for {package.folder_name}\\{filename}."
                )
    return plan


def _patch_metadata(
    metadata: bytes,
    plan: tuple[PackageInstallPlan, ...],
) -> bytes:
    patched = metadata
    for package in plan:
        patched = patch_folder_crc(
            patched,
            package.folder_name,
            package.metadata_crc,
        )
    return patched


def _align_metadata_to_pamt(
    plan: tuple[PackageInstallPlan, ...],
) -> tuple[PackageInstallPlan, ...]:
    return tuple(
        PackageInstallPlan(
            folder_name=package.folder_name,
            source_folder=package.source_folder,
            filenames=package.filenames,
            pamt_crc=package.pamt_crc,
            metadata_crc=package.pamt_crc,
        )
        for package in plan
    )


def _verify_operation(
    source_plan: tuple[PackageInstallPlan, ...],
    game_root: Path,
    expected_metadata: bytes,
) -> int:
    for package in source_plan:
        for filename in package.filenames:
            if _sha256(package.source_folder / filename) != _sha256(
                game_root / package.folder_name / filename
            ):
                raise MetadataFormatError(
                    f"Installed {package.folder_name}\\{filename} does not match the selected package."
                )
        if _pamt_crc(game_root / package.folder_name) != package.pamt_crc:
            raise MetadataFormatError(
                f"Installed {package.folder_name} 0.pamt checksum changed unexpectedly."
            )
    metadata_path = game_root / "meta" / "0.papgt"
    installed_metadata = metadata_path.read_bytes()
    if installed_metadata != expected_metadata:
        raise MetadataFormatError("Installed 0.papgt changed outside permitted fields.")
    index = parse_papgt(installed_metadata)
    for package in source_plan:
        if index.entry_for(package.folder_name).crc != package.metadata_crc:
            raise MetadataFormatError(
                f"Installed metadata has the wrong {package.folder_name} CRC."
            )
    return index.stored_checksum


def install_mod(
    source_path: Path | str,
    game_path: Path | str,
    spec: LanguageSpec,
) -> ModInstallResult:
    plan = _build_install_plan(source_path, spec)
    game_root = _normalize_game_root(game_path, plan)
    for package in plan:
        if package.source_folder.resolve() == (game_root / package.folder_name).resolve():
            raise GamePathError("Install source must be outside the game folder.")

    metadata_path = game_root / "meta" / "0.papgt"
    current_metadata = metadata_path.read_bytes()
    patched_metadata = _patch_metadata(current_metadata, plan)
    backup = _create_backup(game_root, "install", plan)
    try:
        if metadata_path.read_bytes() != current_metadata:
            raise MetadataFormatError(
                "Game 0.papgt changed while installation was starting."
            )
        _assert_game_matches_backup(game_root, backup, plan)
        for package in plan:
            for filename in package.filenames:
                _replace_with_copy(
                    package.source_folder / filename,
                    game_root / package.folder_name / filename,
                )
        _write_metadata_atomically(metadata_path, patched_metadata)
        metadata_checksum = _verify_operation(plan, game_root, patched_metadata)
    except Exception as error:
        try:
            _restore_backup(backup, game_root)
        except Exception as rollback_error:
            raise InstallationError(
                "Installation failed and automatic rollback also failed: "
                f"{rollback_error}. Original error: {error}",
                rolled_back=False,
                backup_directory=backup,
            ) from error
        raise InstallationError(
            f"Installation failed; original files were restored. Cause: {error}",
            rolled_back=True,
            backup_directory=backup,
        ) from error
    return ModInstallResult(
        game_root=game_root,
        metadata_path=metadata_path,
        backup_directory=backup,
        installed_folders=tuple(package.folder_name for package in plan),
        folder_crcs=tuple(
            (package.folder_name, package.metadata_crc) for package in plan
        ),
        metadata_checksum=metadata_checksum,
    )


def _latest_install_backup(game_root: Path) -> Path:
    backup_root = (game_root / "meta" / BACKUP_FOLDER_NAME).resolve()
    if not backup_root.is_dir():
        raise BackupNotFoundError("No install backup exists for this game folder.")
    candidates = sorted(
        (
            path
            for path in backup_root.iterdir()
            if path.is_dir()
            and path.name.startswith("install-")
            and path.resolve().parent == backup_root
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    if not candidates:
        raise BackupNotFoundError("No unused install backup exists for this game folder.")
    return candidates[0]


def _restored_backup_path(install_backup: Path) -> Path:
    suffix = install_backup.name.removeprefix("install-")
    candidate = install_backup.parent / f"restored-{suffix}"
    counter = 1
    while candidate.exists():
        candidate = install_backup.parent / f"restored-{suffix}-{counter}"
        counter += 1
    return candidate


def _legacy_uninstall(game_root: Path) -> ModUninstallResult:
    result = uninstall_0020(game_root)
    return ModUninstallResult(
        game_root=game_root,
        metadata_path=result.metadata_path,
        restored_backup_directory=result.restored_backup_directory,
        safety_backup_directory=result.safety_backup_directory,
        restored_folders=("0020",),
        folder_crcs=(("0020", result.pamt_crc),),
        metadata_checksum=result.metadata_checksum,
    )


def uninstall_mod(game_path: Path | str) -> ModUninstallResult:
    game_root = _normalize_game_root(game_path)
    install_backup = _latest_install_backup(game_root)
    if not (install_backup / MANIFEST_FILENAME).is_file():
        return _legacy_uninstall(game_root)

    restore_plan = _manifest_plan(install_backup)
    verified_plan = _align_metadata_to_pamt(restore_plan)
    _normalize_game_root(game_root, restore_plan)
    metadata_path = game_root / "meta" / "0.papgt"
    current_metadata = metadata_path.read_bytes()
    patched_metadata = _patch_metadata(current_metadata, verified_plan)
    safety_backup = _create_backup(game_root, "uninstall-safety", restore_plan)
    try:
        if metadata_path.read_bytes() != current_metadata:
            raise MetadataFormatError(
                "Game 0.papgt changed while uninstall was starting."
            )
        _assert_game_matches_backup(game_root, safety_backup, restore_plan)
        for package in restore_plan:
            for filename in package.filenames:
                _replace_with_copy(
                    package.source_folder / filename,
                    game_root / package.folder_name / filename,
                )
        _write_metadata_atomically(metadata_path, patched_metadata)
        metadata_checksum = _verify_operation(
            verified_plan,
            game_root,
            patched_metadata,
        )
        restored_backup = _restored_backup_path(install_backup)
        os.replace(install_backup, restored_backup)
    except Exception as error:
        try:
            _restore_backup(safety_backup, game_root)
        except Exception as rollback_error:
            raise UninstallationError(
                "Uninstall failed and automatic rollback also failed: "
                f"{rollback_error}. Original error: {error}",
                rolled_back=False,
                safety_backup_directory=safety_backup,
            ) from error
        raise UninstallationError(
            f"Uninstall failed; installed files were restored. Cause: {error}",
            rolled_back=True,
            safety_backup_directory=safety_backup,
        ) from error
    return ModUninstallResult(
        game_root=game_root,
        metadata_path=metadata_path,
        restored_backup_directory=restored_backup,
        safety_backup_directory=safety_backup,
        restored_folders=tuple(package.folder_name for package in restore_plan),
        folder_crcs=tuple(
            (package.folder_name, package.metadata_crc) for package in verified_plan
        ),
        metadata_checksum=metadata_checksum,
    )
