import struct
from dataclasses import dataclass

from .checksums import pa_checksum


SELF_CHECKSUM_OFFSET = 4
ENTRY_COUNT_OFFSET = 8
ENTRY_TABLE_OFFSET = 12
ENTRY_SIZE = 12


class MetadataFormatError(ValueError):
    """Raised when 0.papgt is damaged or has an unsupported structure."""


@dataclass(frozen=True)
class PapgtEntry:
    name: str
    flags: int
    crc: int
    crc_offset: int


@dataclass(frozen=True)
class PapgtIndex:
    entries: tuple[PapgtEntry, ...]
    stored_checksum: int
    computed_checksum: int

    def entry_for(self, folder_name: str) -> PapgtEntry:
        matches = [entry for entry in self.entries if entry.name == folder_name]
        if len(matches) != 1:
            raise MetadataFormatError(
                f"0.papgt must contain exactly one {folder_name} entry; "
                f"found {len(matches)}."
            )
        return matches[0]


def _require_range(data: bytes | bytearray, offset: int, size: int, field: str) -> None:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise MetadataFormatError(f"0.papgt is truncated while reading {field}.")


def parse_papgt(data: bytes | bytearray) -> PapgtIndex:
    """Parse and validate the current game metadata without changing it."""
    _require_range(data, 0, ENTRY_TABLE_OFFSET, "header")
    stored_checksum = struct.unpack_from("<I", data, SELF_CHECKSUM_OFFSET)[0]
    computed_checksum = pa_checksum(data[ENTRY_TABLE_OFFSET:])
    if stored_checksum != computed_checksum:
        raise MetadataFormatError(
            "0.papgt self-checksum does not match its current contents."
        )

    entry_count = data[ENTRY_COUNT_OFFSET]
    table_size = entry_count * ENTRY_SIZE
    table_end = ENTRY_TABLE_OFFSET + table_size
    _require_range(data, ENTRY_TABLE_OFFSET, table_size, "entry table")
    _require_range(data, table_end, 4, "string table length")
    string_size = struct.unpack_from("<I", data, table_end)[0]
    strings_start = table_end + 4
    strings_end = strings_start + string_size
    _require_range(data, strings_start, string_size, "string table")
    if strings_end != len(data):
        raise MetadataFormatError(
            "0.papgt string table length does not match the file size."
        )

    strings = data[strings_start:strings_end]
    entries: list[PapgtEntry] = []
    names: set[str] = set()
    for index in range(entry_count):
        record_offset = ENTRY_TABLE_OFFSET + index * ENTRY_SIZE
        flags, string_offset, crc = struct.unpack_from("<III", data, record_offset)
        if string_offset >= string_size:
            raise MetadataFormatError(
                f"0.papgt entry {index} has an invalid string offset."
            )
        terminator = strings.find(b"\x00", string_offset)
        if terminator < 0:
            raise MetadataFormatError(
                f"0.papgt entry {index} name is not null-terminated."
            )
        try:
            name = bytes(strings[string_offset:terminator]).decode("utf-8")
        except UnicodeDecodeError as error:
            raise MetadataFormatError(
                f"0.papgt entry {index} name is not valid UTF-8."
            ) from error
        if not name:
            raise MetadataFormatError(f"0.papgt entry {index} has an empty name.")
        if name in names:
            raise MetadataFormatError(f"0.papgt contains duplicate entry name: {name}.")
        names.add(name)
        entries.append(
            PapgtEntry(
                name=name,
                flags=flags,
                crc=crc,
                crc_offset=record_offset + 8,
            )
        )

    return PapgtIndex(tuple(entries), stored_checksum, computed_checksum)


def patch_folder_crc(
    data: bytes | bytearray,
    folder_name: str,
    folder_crc: int,
) -> bytes:
    """Return new metadata with one folder CRC and the self-checksum updated."""
    if not 0 <= folder_crc <= 0xFFFFFFFF:
        raise ValueError("Folder CRC must be an unsigned 32-bit value.")
    index = parse_papgt(data)
    entry = index.entry_for(folder_name)
    patched = bytearray(data)
    struct.pack_into("<I", patched, entry.crc_offset, folder_crc)
    struct.pack_into(
        "<I",
        patched,
        SELF_CHECKSUM_OFFSET,
        pa_checksum(patched[ENTRY_TABLE_OFFSET:]),
    )
    return bytes(patched)


COMPACT_TRAILER_SIZE = 64
COMPACT_ENTRY_TABLE_OFFSET = 4


def is_compact_papgt(data: bytes | bytearray) -> bool:
    """Recognise the game 2.04 ``0.papgt`` layout without interpreting it.

    That layout is ``[u32][N x 12-byte entries][u32 string size][names][64 bytes]``.
    Its checksum scheme is not understood, so callers must leave such a file
    unchanged. The check is strict: every entry must point at a NUL-terminated
    UTF-8 name and the sizes must add up exactly.
    """
    total = len(data)
    if total < COMPACT_ENTRY_TABLE_OFFSET + 4 + COMPACT_TRAILER_SIZE:
        return False
    maximum = (total - COMPACT_ENTRY_TABLE_OFFSET - 4 - COMPACT_TRAILER_SIZE) // ENTRY_SIZE
    for count in range(1, maximum + 1):
        size_offset = COMPACT_ENTRY_TABLE_OFFSET + count * ENTRY_SIZE
        string_size = struct.unpack_from("<I", data, size_offset)[0]
        if size_offset + 4 + string_size + COMPACT_TRAILER_SIZE != total:
            continue
        strings = bytes(data[size_offset + 4 : size_offset + 4 + string_size])
        if not strings.endswith(b"\x00"):
            return False
        names: set[bytes] = set()
        for index in range(count):
            _flags, name_offset, _crc = struct.unpack_from(
                "<III", data, COMPACT_ENTRY_TABLE_OFFSET + index * ENTRY_SIZE
            )
            if name_offset >= string_size:
                return False
            end = strings.find(b"\x00", name_offset)
            name = strings[name_offset:end]
            if end < 0 or not name or name in names:
                return False
            try:
                name.decode("utf-8")
            except UnicodeDecodeError:
                return False
            names.add(name)
        return True
    return False
