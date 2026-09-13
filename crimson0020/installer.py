import hashlib
import os
import shutil
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .archive import extract_english_paloc, parse_pamt, validate_english_0020
from .checksums import pa_checksum
from .metadata import MetadataFormatError, parse_papgt, patch_folder_crc
from .service import normalize_0020_folder


BACKUP_FOLDER_NAME = "crimson0020_backups"


class GamePathError(ValueError):
    """Raised when the selected Crimson Desert game folder is incomplete."""


class BackupNotFoundError(ValueError):
    """Raised when Uninstall has no install backup to restore."""


class BackupFormatError(ValueError):
    """Raised when an install backup is incomplete or inconsistent."""


class InstallationError(RuntimeError):
    """Raised after an install failure, with explicit rollback information."""

    def __init__(
        self,
        message: str,
        *,
        rolled_back: bool,
        backup_directory: Path,
    ) -> None:
        super().__init__(message)
        self.rolled_back = rolled_back
        self.backup_directory = backup_directory


class UninstallationError(RuntimeError):
    """Raised after an uninstall failure, with explicit rollback information."""

    def __init__(
        self,
        message: str,
        *,
        rolled_back: bool,
        safety_backup_directory: Path,
    ) -> None:
        super().__init__(message)
        self.rolled_back = rolled_back
        self.safety_backup_directory = safety_backup_directory


@dataclass(frozen=True)
class InstallResult:
    game_0020: Path
    metadata_path: Path
    backup_directory: Path
    pamt_crc: int
    metadata_checksum: int


@dataclass(frozen=True)
class UninstallResult:
    game_0020: Path
    metadata_path: Path
    restored_backup_directory: Path
    safety_backup_directory: Path
    pamt_crc: int
    metadata_checksum: int


def normalize_game_root(selected_path: Path | str) -> Path:
    value = str(selected_path).strip()
    if not value:
        raise GamePathError("Please select the Crimson Desert game folder.")
    game_root = Path(value).expanduser().resolve()
    metadata = game_root / "meta" / "0.papgt"
    game_0020 = game_root / "0020"
    if not game_root.is_dir() or not metadata.is_file():
        raise GamePathError(
            "The game folder must contain meta\\0.papgt."
        )
    if not game_0020.is_dir():
        raise GamePathError("The game folder must contain the 0020 directory.")
    if not (game_0020 / "0.pamt").is_file() or not (game_0020 / "0.paz").is_file():
        raise GamePathError("The game 0020 folder must contain 0.pamt and 0.paz.")
    return game_root


def _validate_install_package(package_0020: Path) -> int:
    index = parse_pamt(package_0020 / "0.pamt", package_0020)
    validate_english_0020(index)
    extract_english_paloc(package_0020)

    pamt_data = (package_0020 / "0.pamt").read_bytes()
    stored_pamt_checksum = struct.unpack_from("<I", pamt_data, 0)[0]
    computed_pamt_checksum = pa_checksum(pamt_data[12:])
    if stored_pamt_checksum != computed_pamt_checksum:
        raise MetadataFormatError("Source 0.pamt header checksum does not match.")

    record = index.paz_records[0]
    paz_data = (package_0020 / "0.paz").read_bytes()
    if record.stored_size != len(paz_data):
        raise MetadataFormatError("Source 0.paz size does not match 0.pamt.")
    if record.stored_checksum != pa_checksum(paz_data):
        raise MetadataFormatError("Source 0.paz checksum does not match 0.pamt.")
    return computed_pamt_checksum


def _unique_backup_directory(backup_root: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime(f"{prefix}-%Y%m%d-%H%M%S-%f")
    candidate = backup_root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = backup_root / f"{timestamp}-{suffix}"
        suffix += 1
    return candidate


def _remove_backup_staging(
    staging: Path,
    backup_root: Path,
    prefix: str,
) -> None:
    resolved_staging = staging.resolve()
    resolved_root = backup_root.resolve()
    if (
        resolved_staging.parent != resolved_root
        or not staging.name.startswith(f".{prefix}-")
        or not staging.name.endswith(".tmp")
    ):
        raise RuntimeError("Refusing to remove an unexpected backup staging path.")
    if staging.exists():
        shutil.rmtree(staging)


def _create_backup(game_root: Path, prefix: str) -> Path:
    backup_root = game_root / "meta" / BACKUP_FOLDER_NAME
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = _unique_backup_directory(backup_root, prefix)
    staging = backup_root / f".{destination.name}-{uuid.uuid4().hex}.tmp"
    try:
        (staging / "0020").mkdir(parents=True)
        (staging / "meta").mkdir()
        shutil.copy2(game_root / "0020" / "0.pamt", staging / "0020" / "0.pamt")
        shutil.copy2(game_root / "0020" / "0.paz", staging / "0020" / "0.paz")
        shutil.copy2(game_root / "meta" / "0.papgt", staging / "meta" / "0.papgt")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_install(
    source_0020: Path,
    game_root: Path,
    expected_metadata: bytes,
    expected_pamt_crc: int,
) -> int:
    game_0020 = game_root / "0020"
    for filename in ("0.pamt", "0.paz"):
        if _sha256(source_0020 / filename) != _sha256(game_0020 / filename):
            raise MetadataFormatError(
                f"Installed {filename} does not match the selected package."
            )

    installed_pamt_crc = _validate_install_package(game_0020)
    if installed_pamt_crc != expected_pamt_crc:
        raise MetadataFormatError("Installed 0.pamt checksum changed unexpectedly.")

    metadata_path = game_root / "meta" / "0.papgt"
    installed_metadata = metadata_path.read_bytes()
    if installed_metadata != expected_metadata:
        raise MetadataFormatError(
            "Installed 0.papgt changed outside the permitted 0020 fields."
        )
    metadata_index = parse_papgt(installed_metadata)
    if metadata_index.entry_for("0020").crc != expected_pamt_crc:
        raise MetadataFormatError("Installed 0.papgt has the wrong 0020 CRC.")
    return metadata_index.stored_checksum


def _restore_backup(backup: Path, game_root: Path) -> None:
    _replace_with_copy(backup / "0020" / "0.paz", game_root / "0020" / "0.paz")
    _replace_with_copy(backup / "0020" / "0.pamt", game_root / "0020" / "0.pamt")
    _replace_with_copy(backup / "meta" / "0.papgt", game_root / "meta" / "0.papgt")
    pairs = (
        (backup / "0020" / "0.paz", game_root / "0020" / "0.paz"),
        (backup / "0020" / "0.pamt", game_root / "0020" / "0.pamt"),
        (backup / "meta" / "0.papgt", game_root / "meta" / "0.papgt"),
    )
    for source, restored in pairs:
        if _sha256(source) != _sha256(restored):
            raise OSError(f"Rollback verification failed for {restored}.")


def _latest_install_backup(game_root: Path) -> tuple[Path, int]:
    backup_root = (game_root / "meta" / BACKUP_FOLDER_NAME).resolve()
    if not backup_root.is_dir():
        raise BackupNotFoundError(
            "No install backup exists for this game folder."
        )
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
        raise BackupNotFoundError(
            "No unused install backup exists for this game folder."
        )

    backup = candidates[0]
    package = backup / "0020"
    metadata_path = backup / "meta" / "0.papgt"
    if not metadata_path.is_file():
        raise BackupFormatError(
            f"Install backup is missing meta\\0.papgt: {backup}"
        )
    try:
        package = normalize_0020_folder(package)
        pamt_crc = _validate_install_package(package)
        backup_metadata = metadata_path.read_bytes()
        metadata_crc = parse_papgt(backup_metadata).entry_for("0020").crc
    except Exception as error:
        raise BackupFormatError(f"Install backup is invalid: {backup}") from error
    if metadata_crc != pamt_crc:
        raise BackupFormatError(
            "Install backup 0.papgt does not match its restored 0.pamt."
        )
    return backup, pamt_crc


def _restored_backup_path(install_backup: Path) -> Path:
    backup_root = install_backup.parent
    suffix = install_backup.name.removeprefix("install-")
    candidate = backup_root / f"restored-{suffix}"
    counter = 1
    while candidate.exists():
        candidate = backup_root / f"restored-{suffix}-{counter}"
        counter += 1
    return candidate


def install_0020(source_0020: Path | str, game_path: Path | str) -> InstallResult:
    """Install 0020 and patch only its CRC in the game's current 0.papgt."""
    source = normalize_0020_folder(source_0020)
    game_root = normalize_game_root(game_path)
    game_0020 = game_root / "0020"
    if source == game_0020:
        raise GamePathError("Install source must be outside the game 0020 folder.")

    pamt_crc = _validate_install_package(source)
    metadata_path = game_root / "meta" / "0.papgt"
    current_metadata = metadata_path.read_bytes()
    parse_papgt(current_metadata).entry_for("0020")
    patched_metadata = patch_folder_crc(current_metadata, "0020", pamt_crc)

    backup = _create_backup(game_root, "install")
    try:
        if metadata_path.read_bytes() != current_metadata:
            raise MetadataFormatError(
                "Game 0.papgt changed while installation was starting."
            )
        _replace_with_copy(source / "0.paz", game_0020 / "0.paz")
        _replace_with_copy(source / "0.pamt", game_0020 / "0.pamt")
        _write_metadata_atomically(metadata_path, patched_metadata)
        metadata_checksum = _verify_install(
            source,
            game_root,
            patched_metadata,
            pamt_crc,
        )
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

    return InstallResult(
        game_0020=game_0020,
        metadata_path=metadata_path,
        backup_directory=backup,
        pamt_crc=pamt_crc,
        metadata_checksum=metadata_checksum,
    )


def uninstall_0020(game_path: Path | str) -> UninstallResult:
    """Restore the latest 0020 backup while preserving current unrelated metadata."""
    game_root = normalize_game_root(game_path)
    game_0020 = game_root / "0020"
    restore_backup, restored_pamt_crc = _latest_install_backup(game_root)
    restore_package = restore_backup / "0020"

    metadata_path = game_root / "meta" / "0.papgt"
    current_metadata = metadata_path.read_bytes()
    parse_papgt(current_metadata).entry_for("0020")
    patched_metadata = patch_folder_crc(
        current_metadata,
        "0020",
        restored_pamt_crc,
    )

    safety_backup = _create_backup(game_root, "uninstall-safety")
    try:
        if metadata_path.read_bytes() != current_metadata:
            raise MetadataFormatError(
                "Game 0.papgt changed while uninstall was starting."
            )
        _replace_with_copy(restore_package / "0.paz", game_0020 / "0.paz")
        _replace_with_copy(restore_package / "0.pamt", game_0020 / "0.pamt")
        _write_metadata_atomically(metadata_path, patched_metadata)
        metadata_checksum = _verify_install(
            restore_package,
            game_root,
            patched_metadata,
            restored_pamt_crc,
        )
        restored_backup = _restored_backup_path(restore_backup)
        os.replace(restore_backup, restored_backup)
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

    return UninstallResult(
        game_0020=game_0020,
        metadata_path=metadata_path,
        restored_backup_directory=restored_backup,
        safety_backup_directory=safety_backup,
        pamt_crc=restored_pamt_crc,
        metadata_checksum=metadata_checksum,
    )
