import os
import struct
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .checksums import pa_checksum, pa_checksum_file
from .crypto import compress_lz4, decrypt, decompress_lz4, encrypt
from .languages import LanguageSpec, get_language_spec


TARGET_ENTRY_PATH = "gamedata/localizationstring_eng.paloc"
TARGET_FILENAME = "localizationstring_eng.paloc"


class ArchiveFormatError(ValueError):
    """Raised when the selected 0020 package does not match the supported layout."""


@dataclass(frozen=True)
class PazRecord:
    index: int
    checksum_offset: int
    size_offset: int
    stored_checksum: int
    stored_size: int


@dataclass(frozen=True)
class PamtEntry:
    path: str
    paz_path: Path
    offset: int
    comp_size: int
    orig_size: int
    flags: int
    paz_index: int
    record_offset: int

    @property
    def compression_type(self) -> int:
        return (self.flags >> 16) & 0x0F


@dataclass(frozen=True)
class PamtIndex:
    pamt_path: Path
    paz_records: tuple[PazRecord, ...]
    entries: tuple[PamtEntry, ...]

    def find_entry(self, target_path: str) -> PamtEntry:
        target = target_path.replace("\\", "/").lower()
        for entry in self.entries:
            if entry.path.lower() == target:
                return entry
        raise ArchiveFormatError(f"Entry not found in 0.pamt: {target_path}")


@dataclass(frozen=True)
class RepackStats:
    original_size: int
    compressed_size: int
    paz_size: int
    paz_checksum: int
    pamt_checksum: int


@dataclass(frozen=True)
class LocalizationPaloc:
    path: str
    data: bytes


@dataclass(frozen=True)
class AppendStats:
    entry_count: int
    paz_filename: str
    paz_size: int
    paz_checksum: int
    pamt_checksum: int


def _require_range(data: bytes | bytearray, offset: int, size: int, field: str) -> None:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ArchiveFormatError(f"0.pamt is truncated while reading {field}.")


def _read_uint32(data: bytes, offset: int, field: str) -> int:
    _require_range(data, offset, 4, field)
    return struct.unpack_from("<I", data, offset)[0]


def _read_named_nodes(data: bytes, start: int, size: int) -> dict[int, tuple[int, str]]:
    end = start + size
    _require_range(data, start, size, "path nodes")
    nodes: dict[int, tuple[int, str]] = {}
    offset = start
    while offset < end:
        relative_offset = offset - start
        _require_range(data, offset, 5, "path node")
        parent = struct.unpack_from("<I", data, offset)[0]
        name_length = data[offset + 4]
        _require_range(data, offset + 5, name_length, "path node name")
        try:
            name = data[offset + 5 : offset + 5 + name_length].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArchiveFormatError(f"Invalid UTF-8 in 0.pamt path node: {error}.") from error
        nodes[relative_offset] = (parent, name)
        offset += 5 + name_length
    if offset != end:
        raise ArchiveFormatError("0.pamt path node section has an invalid length.")
    return nodes


def _read_compact_nodes(data: bytes, start: int, size: int) -> dict[int, tuple[int, str]]:
    """Read the newer path-node layout used by recent game patches.

    Each node is ``[u8 parent_distance][u8 name_length][name]`` where
    ``parent_distance`` is the number of bytes between this node and its parent
    node (0 means "no parent"). Names are prefix fragments, so a full path is the
    concatenation of the whole parent chain.
    """
    end = start + size
    _require_range(data, start, size, "path nodes")
    nodes: dict[int, tuple[int, str]] = {}
    offset = start
    while offset < end:
        relative_offset = offset - start
        _require_range(data, offset, 2, "path node")
        distance = data[offset]
        name_length = data[offset + 1]
        _require_range(data, offset + 2, name_length, "path node name")
        if distance == 0:
            parent = 0xFFFFFFFF
        else:
            parent = relative_offset - distance
            if parent < 0 or parent not in nodes:
                raise ArchiveFormatError("0.pamt path node references an unknown parent.")
        try:
            name = data[offset + 2 : offset + 2 + name_length].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArchiveFormatError(f"Invalid UTF-8 in 0.pamt path node: {error}.") from error
        nodes[relative_offset] = (parent, name)
        offset += 2 + name_length
    if offset != end:
        raise ArchiveFormatError("0.pamt path node section has an invalid length.")
    return nodes


def _build_node_path(nodes: dict[int, tuple[int, str]], node_reference: int) -> str:
    parts: list[str] = []
    current = node_reference
    visited: set[int] = set()
    while current != 0xFFFFFFFF:
        if current in visited or len(visited) >= 64:
            raise ArchiveFormatError("0.pamt path tree contains a cycle or is too deep.")
        if current not in nodes:
            raise ArchiveFormatError("0.pamt file entry references an unknown path node.")
        visited.add(current)
        parent, name = nodes[current]
        parts.append(name)
        current = parent
    return "".join(reversed(parts))


def parse_pamt(pamt_path: Path, paz_directory: Path) -> PamtIndex:
    try:
        data = pamt_path.read_bytes()
    except OSError as error:
        raise ArchiveFormatError(f"Cannot read 0.pamt: {error}.") from error
    if len(data) < 24:
        raise ArchiveFormatError("0.pamt is too small.")

    paz_count = _read_uint32(data, 4, "PAZ count")
    if paz_count == 0 or paz_count > 4096:
        raise ArchiveFormatError(f"Unsupported PAZ count: {paz_count}.")

    offset = 16
    paz_records: list[PazRecord] = []
    for index in range(paz_count):
        _require_range(data, offset, 8, "PAZ table")
        paz_records.append(
            PazRecord(
                index=index,
                checksum_offset=offset,
                size_offset=offset + 4,
                stored_checksum=struct.unpack_from("<I", data, offset)[0],
                stored_size=struct.unpack_from("<I", data, offset + 4)[0],
            )
        )
        offset += 8
        if index < paz_count - 1:
            _require_range(data, offset, 4, "PAZ table separator")
            offset += 4

    folder_size = _read_uint32(data, offset, "folder section length")
    offset += 4
    # Older packages store each node as [u32 parent][u8 length][name] with
    # 0xFFFFFFFF marking the root. Newer packages use the compact layout.
    compact = _read_uint32(data, offset, "folder node") != 0xFFFFFFFF
    read_nodes = _read_compact_nodes if compact else _read_named_nodes
    folder_nodes = read_nodes(data, offset, folder_size)
    offset += folder_size
    root_names = [name for parent, name in folder_nodes.values() if parent == 0xFFFFFFFF]
    folder_prefix = root_names[-1] if root_names else ""

    node_size = _read_uint32(data, offset, "node section length")
    offset += 4
    nodes = read_nodes(data, offset, node_size)
    offset += node_size

    folder_count = _read_uint32(data, offset, "folder record count")
    offset += 8
    folder_record_size = folder_count * 16
    _require_range(data, offset, folder_record_size, "folder records")
    offset += folder_record_size

    # The u32 just before the file records is the file count.
    records_end = len(data)
    extensions: list[str] = []
    if (len(data) - offset) % 20 != 0:
        # Newer packages append an extension table after the file records:
        # [u32 size][NUL separated strings]; flags >> 24 selects the extension.
        file_count = _read_uint32(data, offset - 4, "file record count")
        records_end = offset + file_count * 20
        _require_range(data, offset, file_count * 20, "file records")
        table_size = _read_uint32(data, records_end, "extension table length")
        if table_size != len(data) - records_end - 4:
            raise ArchiveFormatError("0.pamt file record section has an invalid length.")
        try:
            extensions = data[records_end + 4 :].decode("utf-8").split("\0")
        except UnicodeDecodeError as error:
            raise ArchiveFormatError(
                f"Invalid UTF-8 in 0.pamt extension table: {error}."
            ) from error
    if records_end == offset:
        raise ArchiveFormatError("0.pamt file record section has an invalid length.")

    entries: list[PamtEntry] = []
    pamt_base_index = int(pamt_path.stem)
    while offset < records_end:
        record_offset = offset
        node_reference, paz_offset, comp_size, orig_size, flags = struct.unpack_from(
            "<IIIII", data, offset
        )
        offset += 20
        paz_index = flags & 0xFF
        if paz_index >= paz_count:
            raise ArchiveFormatError(f"Entry references invalid PAZ index {paz_index}.")
        node_path = _build_node_path(nodes, node_reference)
        extension_index = flags >> 24
        if extension_index:
            if extension_index >= len(extensions):
                raise ArchiveFormatError(
                    f"Entry references invalid extension index {extension_index}."
                )
            node_path += extensions[extension_index]
        full_path = f"{folder_prefix}/{node_path}" if folder_prefix else node_path
        paz_number = pamt_base_index + paz_index
        entries.append(
            PamtEntry(
                path=full_path,
                paz_path=paz_directory / f"{paz_number}.paz",
                offset=paz_offset,
                comp_size=comp_size,
                orig_size=orig_size,
                flags=flags,
                paz_index=paz_index,
                record_offset=record_offset,
            )
        )

    return PamtIndex(pamt_path, tuple(paz_records), tuple(entries))


def validate_localization_package(
    index: PamtIndex,
    spec: LanguageSpec,
) -> tuple[PamtEntry, ...]:
    if not index.entries:
        raise ArchiveFormatError(
            f"The {spec.package_folder} package does not contain localization entries."
        )
    if len(index.paz_records) != 1:
        raise ArchiveFormatError(
            f"The {spec.package_folder} package must contain exactly one PAZ record; "
            f"found {len(index.paz_records)}."
        )

    normalized_paths = [entry.path.replace("\\", "/").lower() for entry in index.entries]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ArchiveFormatError("The localization package contains duplicate entry paths.")

    if len(index.entries) == 1:
        entry = index.find_entry(spec.paloc_entry_path)
        if entry.offset != 0 or entry.paz_index != 0:
            raise ArchiveFormatError("The localization entry must be first in 0.paz.")
    elif any(
        not path.startswith("gamedata/") or not path.endswith(".paloc")
        for path in normalized_paths
    ):
        raise ArchiveFormatError(
            f"The {spec.package_folder} package contains non-localization entries."
        )

    for entry in index.entries:
        if entry.paz_index != 0:
            raise ArchiveFormatError(
                f"Localization entry {entry.path} references an unsupported PAZ index."
            )
        allowed_compression = (2,) if len(index.entries) == 1 else (0, 2)
        if entry.compression_type not in allowed_compression:
            if len(index.entries) == 1:
                raise ArchiveFormatError(
                    "The localization entry must use LZ4 compression type 2; "
                    f"found {entry.compression_type}."
                )
            raise ArchiveFormatError(
                f"Unsupported compression type {entry.compression_type} for {entry.path}."
            )
        if not entry.paz_path.is_file():
            raise ArchiveFormatError(f"Missing PAZ file: {entry.paz_path}")
        if entry.offset + entry.comp_size > entry.paz_path.stat().st_size:
            raise ArchiveFormatError(f"Archive entry extends beyond {entry.paz_path.name}.")
        if entry.offset % 16:
            raise ArchiveFormatError(
                f"Localization entry {entry.path} has a non-aligned PAZ offset."
            )
    ordered_entries = sorted(index.entries, key=lambda entry: entry.offset)
    for previous, current in zip(ordered_entries, ordered_entries[1:], strict=False):
        if previous.offset + previous.comp_size > current.offset:
            raise ArchiveFormatError("The localization package contains overlapping entries.")
    _validate_localization_package_integrity(index)
    return index.entries


def _validate_localization_package_integrity(index: PamtIndex) -> None:
    try:
        pamt_data = index.pamt_path.read_bytes()
    except OSError as error:
        raise ArchiveFormatError(f"Cannot read 0.pamt: {error}.") from error
    stored_pamt_checksum = int.from_bytes(pamt_data[:4], "little")
    if pa_checksum(pamt_data[12:]) != stored_pamt_checksum:
        raise ArchiveFormatError("0.pamt header checksum verification failed.")

    paz_record = index.paz_records[0]
    paz_path = index.entries[0].paz_path
    try:
        paz_size = paz_path.stat().st_size
        paz_checksum = pa_checksum_file(paz_path)
    except OSError as error:
        raise ArchiveFormatError(f"Cannot verify {paz_path.name}: {error}.") from error
    if paz_size != paz_record.stored_size:
        raise ArchiveFormatError(
            f"PAZ size in 0.pamt does not match {paz_path.name}."
        )
    if paz_checksum != paz_record.stored_checksum:
        raise ArchiveFormatError(
            f"PAZ checksum in 0.pamt does not match {paz_path.name}."
        )


def validate_english_0020(index: PamtIndex) -> tuple[PamtEntry, ...]:
    return validate_localization_package(index, get_language_spec("English"))


def _entry_is_encrypted(entry_path: str) -> bool:
    lowered = entry_path.lower()
    return (
        lowered.endswith(".xml")
        or lowered.endswith(".paloc")
        or lowered.endswith("cdcommon_font_eng.css")
    )


def _read_entry_payload(entry: PamtEntry) -> bytes:
    if not entry.paz_path.is_file():
        raise ArchiveFormatError(f"Missing PAZ file: {entry.paz_path}")
    file_size = entry.paz_path.stat().st_size
    if entry.offset + entry.comp_size > file_size:
        raise ArchiveFormatError(f"Archive entry extends beyond {entry.paz_path.name}.")
    with entry.paz_path.open("rb") as stream:
        stream.seek(entry.offset)
        payload = stream.read(entry.comp_size)
    if len(payload) != entry.comp_size:
        raise ArchiveFormatError(f"Cannot read the complete entry from {entry.paz_path.name}.")
    return payload


def extract_archive_entry(package_folder: Path, target_path: str) -> bytes:
    index = parse_pamt(package_folder / "0.pamt", package_folder)
    entry = index.find_entry(target_path)
    payload = _read_entry_payload(entry)
    if _entry_is_encrypted(entry.path):
        payload = decrypt(payload, Path(entry.path).name)
    if entry.compression_type == 2:
        try:
            payload = decompress_lz4(payload, entry.orig_size)
        except Exception as error:
            raise ArchiveFormatError(
                f"Cannot decompress archive entry {entry.path}: {error}."
            ) from error
    elif entry.compression_type != 0:
        raise ArchiveFormatError(
            f"Unsupported compression type {entry.compression_type} for {entry.path}."
        )
    if len(payload) != entry.orig_size:
        raise ArchiveFormatError(
            f"Extracted size for {entry.path} does not match 0.pamt."
        )
    return payload


def extract_localization_paloc(package_folder: Path, spec: LanguageSpec) -> bytes:
    index = parse_pamt(package_folder / "0.pamt", package_folder)
    entries = validate_localization_package(index, spec)
    if len(entries) != 1:
        raise ArchiveFormatError(
            "This operation requires a legacy single-PALOC localization package."
        )
    try:
        return extract_archive_entry(package_folder, spec.paloc_entry_path)
    except Exception as error:
        raise ArchiveFormatError(
            f"Cannot decompress the localization entry: {error}."
        ) from error


def extract_english_paloc(package_0020: Path) -> bytes:
    return extract_localization_paloc(package_0020, get_language_spec("English"))


def extract_localization_palocs(
    package_folder: Path,
    spec: LanguageSpec,
) -> tuple[LocalizationPaloc, ...]:
    index = parse_pamt(package_folder / "0.pamt", package_folder)
    entries = validate_localization_package(index, spec)
    return tuple(
        LocalizationPaloc(entry.path, extract_archive_entry(package_folder, entry.path))
        for entry in entries
    )


def _atomic_write_bytes(path: Path, data: bytes | bytearray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def repack_localization_paloc(
    package_folder: Path,
    paloc_data: bytes,
    spec: LanguageSpec,
) -> RepackStats:
    index = parse_pamt(package_folder / "0.pamt", package_folder)
    entries = validate_localization_package(index, spec)
    if len(entries) != 1:
        raise ArchiveFormatError(
            "This operation requires a legacy single-PALOC localization package."
        )
    entry = entries[0]
    compressed = compress_lz4(paloc_data)
    encrypted = encrypt(compressed, spec.paloc_filename)
    padding_size = (16 - (len(encrypted) % 16)) % 16
    paz_data = encrypted + (b"\x00" * padding_size)
    paz_checksum = pa_checksum(paz_data)

    pamt_data = bytearray(index.pamt_path.read_bytes())
    struct.pack_into("<I", pamt_data, entry.record_offset + 4, 0)
    struct.pack_into("<I", pamt_data, entry.record_offset + 8, len(encrypted))
    struct.pack_into("<I", pamt_data, entry.record_offset + 12, len(paloc_data))
    paz_record = index.paz_records[entry.paz_index]
    struct.pack_into("<I", pamt_data, paz_record.checksum_offset, paz_checksum)
    struct.pack_into("<I", pamt_data, paz_record.size_offset, len(paz_data))
    pamt_checksum = pa_checksum(pamt_data[12:])
    struct.pack_into("<I", pamt_data, 0, pamt_checksum)

    _atomic_write_bytes(entry.paz_path, paz_data)
    _atomic_write_bytes(index.pamt_path, pamt_data)
    return RepackStats(
        original_size=len(paloc_data),
        compressed_size=len(encrypted),
        paz_size=len(paz_data),
        paz_checksum=paz_checksum,
        pamt_checksum=pamt_checksum,
    )


def repack_english_paloc(package_0020: Path, paloc_data: bytes) -> RepackStats:
    return repack_localization_paloc(
        package_0020,
        paloc_data,
        get_language_spec("English"),
    )


def _encode_entry(entry: PamtEntry, plaintext: bytes) -> bytes:
    if entry.compression_type == 2:
        payload = compress_lz4(plaintext)
    elif entry.compression_type == 0:
        payload = plaintext
    else:
        raise ArchiveFormatError(
            f"Unsupported compression type {entry.compression_type} for {entry.path}."
        )
    if _entry_is_encrypted(entry.path):
        payload = encrypt(payload, Path(entry.path).name)
    return payload


def repack_localization_palocs(
    package_folder: Path,
    replacements: Mapping[str, bytes],
    spec: LanguageSpec,
) -> RepackStats:
    index = parse_pamt(package_folder / "0.pamt", package_folder)
    entries = validate_localization_package(index, spec)
    normalized_replacements = {
        path.replace("\\", "/").lower(): data for path, data in replacements.items()
    }
    expected_paths = {entry.path.replace("\\", "/").lower() for entry in entries}
    if set(normalized_replacements) != expected_paths:
        missing = sorted(expected_paths - set(normalized_replacements))
        extra = sorted(set(normalized_replacements) - expected_paths)
        raise ArchiveFormatError(
            "PALOC replacement paths do not match the source package: "
            f"missing={missing}, extra={extra}."
        )

    source_paz = entries[0].paz_path
    rebuilt_paz = bytearray()
    pamt_data = bytearray(index.pamt_path.read_bytes())
    total_original_size = 0
    total_compressed_size = 0

    for entry in entries:
        aligned_offset = (len(rebuilt_paz) + 15) & ~15
        rebuilt_paz += b"\x00" * (aligned_offset - len(rebuilt_paz))
        plaintext = normalized_replacements[entry.path.replace("\\", "/").lower()]
        payload = _encode_entry(entry, plaintext)
        rebuilt_paz += payload
        total_original_size += len(plaintext)
        total_compressed_size += len(payload)
        struct.pack_into("<I", pamt_data, entry.record_offset + 4, aligned_offset)
        struct.pack_into("<I", pamt_data, entry.record_offset + 8, len(payload))
        struct.pack_into("<I", pamt_data, entry.record_offset + 12, len(plaintext))

    rebuilt_paz += b"\x00" * ((16 - len(rebuilt_paz) % 16) % 16)
    paz_checksum = pa_checksum(rebuilt_paz)
    paz_record = index.paz_records[0]
    struct.pack_into("<I", pamt_data, paz_record.checksum_offset, paz_checksum)
    struct.pack_into("<I", pamt_data, paz_record.size_offset, len(rebuilt_paz))
    pamt_checksum = pa_checksum(pamt_data[12:])
    struct.pack_into("<I", pamt_data, 0, pamt_checksum)

    _atomic_write_bytes(source_paz, rebuilt_paz)
    _atomic_write_bytes(index.pamt_path, pamt_data)
    return RepackStats(
        original_size=total_original_size,
        compressed_size=total_compressed_size,
        paz_size=len(rebuilt_paz),
        paz_checksum=paz_checksum,
        pamt_checksum=pamt_checksum,
    )


def append_entries_to_last_paz(
    package_folder: Path,
    replacements: dict[str, bytes],
) -> AppendStats:
    if not replacements:
        raise ValueError("At least one archive entry replacement is required.")
    index = parse_pamt(package_folder / "0.pamt", package_folder)
    entries = [(index.find_entry(path), data) for path, data in replacements.items()]
    last_index = len(index.paz_records) - 1
    last_paz_number = int(index.pamt_path.stem) + last_index
    last_paz_path = package_folder / f"{last_paz_number}.paz"
    if not last_paz_path.is_file():
        raise ArchiveFormatError(f"Missing last PAZ file: {last_paz_path}")

    pamt_data = bytearray(index.pamt_path.read_bytes())
    with last_paz_path.open("ab") as stream:
        for entry, plaintext in entries:
            current_size = stream.tell()
            aligned_offset = (current_size + 15) & ~15
            if aligned_offset > current_size:
                stream.write(b"\x00" * (aligned_offset - current_size))
            payload = _encode_entry(entry, plaintext)
            stream.write(payload)
            padding_size = (16 - len(payload) % 16) % 16
            if padding_size:
                stream.write(b"\x00" * padding_size)
            flags = (entry.flags & 0xFFFF0000) | last_index
            struct.pack_into("<I", pamt_data, entry.record_offset + 4, aligned_offset)
            struct.pack_into("<I", pamt_data, entry.record_offset + 8, len(payload))
            struct.pack_into("<I", pamt_data, entry.record_offset + 12, len(plaintext))
            struct.pack_into("<I", pamt_data, entry.record_offset + 16, flags)

    paz_size = last_paz_path.stat().st_size
    paz_checksum = pa_checksum_file(last_paz_path)
    record = index.paz_records[last_index]
    struct.pack_into("<I", pamt_data, record.checksum_offset, paz_checksum)
    struct.pack_into("<I", pamt_data, record.size_offset, paz_size)
    pamt_checksum = pa_checksum(pamt_data[12:])
    struct.pack_into("<I", pamt_data, 0, pamt_checksum)
    _atomic_write_bytes(index.pamt_path, pamt_data)
    return AppendStats(
        entry_count=len(entries),
        paz_filename=last_paz_path.name,
        paz_size=paz_size,
        paz_checksum=paz_checksum,
        pamt_checksum=pamt_checksum,
    )
