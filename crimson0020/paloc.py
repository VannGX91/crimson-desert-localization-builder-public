import json
import os
import struct
import uuid
from pathlib import Path
from typing import Any

from .crypto import compress_lz4, decompress_lz4


MARKER_SIZE = 8
UINT32_MAX = 0xFFFFFFFF
PALOC_LZ4_MAGIC = b"paloc\x00\x00\x00\x00"
PALOC_LZ4_HEADER_SIZE = 512
PALOC_LZ4_STORAGE_FORMAT = "lz4-container-v1"


class PalocFormatError(ValueError):
    """Raised when PALOC bytes or localization JSON are malformed."""


def _read_uint32(data: bytes, offset: int, field_name: str) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise PalocFormatError(f"PALOC ended while reading {field_name}.")
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def _decode_utf8(value: bytes, field_name: str) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PalocFormatError(f"Invalid UTF-8 in {field_name}: {error}.") from error


def _unwrap_paloc(data: bytes) -> tuple[bytes, str | None]:
    if not data.startswith(PALOC_LZ4_MAGIC):
        return data, None
    if len(data) < PALOC_LZ4_HEADER_SIZE:
        raise PalocFormatError("PALOC LZ4 container header is truncated.")

    stored_size, original_size = struct.unpack_from("<II", data, len(PALOC_LZ4_MAGIC))
    if any(data[17:PALOC_LZ4_HEADER_SIZE]):
        raise PalocFormatError("PALOC LZ4 container has nonzero reserved header bytes.")

    payload = data[PALOC_LZ4_HEADER_SIZE:]
    if stored_size != len(payload):
        raise PalocFormatError(
            "PALOC LZ4 container stored size does not match the payload."
        )
    try:
        raw_paloc = decompress_lz4(payload, original_size)
    except Exception as error:
        raise PalocFormatError(
            f"Cannot decompress PALOC LZ4 container: {error}."
        ) from error
    if len(raw_paloc) != original_size:
        raise PalocFormatError(
            "PALOC LZ4 container original size does not match the decoded payload."
        )
    return raw_paloc, PALOC_LZ4_STORAGE_FORMAT


def _wrap_paloc(data: bytes, storage_format: str | None) -> bytes:
    if storage_format is None:
        return data
    if storage_format != PALOC_LZ4_STORAGE_FORMAT:
        raise PalocFormatError("Unsupported metadata.storage_format.")

    payload = compress_lz4(data)
    if len(payload) > UINT32_MAX or len(data) > UINT32_MAX:
        raise PalocFormatError("PALOC LZ4 container is too large.")
    header = PALOC_LZ4_MAGIC + struct.pack("<II", len(payload), len(data))
    return header + (b"\x00" * (PALOC_LZ4_HEADER_SIZE - len(header))) + payload


def paloc_to_json_data(data: bytes) -> dict[str, Any]:
    raw_data, storage_format = _unwrap_paloc(data)
    entries: list[dict[str, str]] = []
    tail_data = b""
    offset = 0

    while offset < len(raw_data):
        record_start = offset
        if len(raw_data) - offset < MARKER_SIZE:
            tail_data = raw_data[offset:]
            break

        marker = raw_data[offset : offset + MARKER_SIZE]
        offset += MARKER_SIZE
        if len(raw_data) - offset < 4:
            tail_data = raw_data[record_start:]
            break

        key_length, offset = _read_uint32(raw_data, offset, "key length")
        if key_length > len(raw_data) - offset:
            raise PalocFormatError("PALOC key length exceeds the remaining data.")
        key_bytes = raw_data[offset : offset + key_length]
        offset += key_length

        value_length, offset = _read_uint32(raw_data, offset, "text length")
        if value_length > len(raw_data) - offset:
            raise PalocFormatError("PALOC text length exceeds the remaining data.")
        value_bytes = raw_data[offset : offset + value_length]
        offset += value_length

        key = _decode_utf8(key_bytes, "key")
        value = _decode_utf8(value_bytes, f"text for key {key!r}")
        entries.append(
            {
                "marker": marker.hex(),
                "key": key,
                "original": value,
                "translation": value,
            }
        )

    metadata = {"tail_bytes": tail_data.hex()}
    if storage_format is not None:
        metadata["storage_format"] = storage_format
    return {
        "metadata": metadata,
        "entries": entries,
    }


def _validate_hex(value: Any, field_name: str, byte_length: int | None = None) -> bytes:
    if not isinstance(value, str):
        raise PalocFormatError(f"{field_name} must be a hexadecimal string.")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise PalocFormatError(f"Invalid hexadecimal value in {field_name}.") from error
    if byte_length is not None and len(decoded) != byte_length:
        raise PalocFormatError(f"{field_name} must contain exactly {byte_length} bytes.")
    return decoded


def _encode_text(value: Any, field_name: str) -> bytes:
    if not isinstance(value, str):
        raise PalocFormatError(f"{field_name} must be text.")
    encoded = value.encode("utf-8")
    if len(encoded) > UINT32_MAX:
        raise PalocFormatError(f"{field_name} is too large for PALOC.")
    return encoded


def json_data_to_paloc(data: Any) -> bytes:
    if not isinstance(data, dict):
        raise PalocFormatError("JSON root must be an object.")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PalocFormatError("JSON must contain a non-empty entries list.")

    output = bytearray()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PalocFormatError(f"entries[{index}] must be an object.")
        marker = _validate_hex(entry.get("marker"), f"entries[{index}].marker", MARKER_SIZE)
        key = _encode_text(entry.get("key"), f"entries[{index}].key")
        translation = _encode_text(
            entry.get("translation"),
            f"entries[{index}].translation",
        )
        output += marker
        output += struct.pack("<I", len(key))
        output += key
        output += struct.pack("<I", len(translation))
        output += translation

    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise PalocFormatError("metadata must be an object.")
    output += _validate_hex(metadata.get("tail_bytes", ""), "metadata.tail_bytes")
    storage_format = metadata.get("storage_format")
    if storage_format is not None and not isinstance(storage_format, str):
        raise PalocFormatError("metadata.storage_format must be text.")
    return _wrap_paloc(bytes(output), storage_format)


def read_paloc_json(json_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PalocFormatError(f"Cannot read localization JSON: {error}.") from error
    if not isinstance(data, dict):
        raise PalocFormatError("JSON root must be an object.")
    return data


def load_json_paloc(json_path: Path) -> tuple[bytes, int]:
    data = read_paloc_json(json_path)
    paloc = json_data_to_paloc(data)
    return paloc, len(data["entries"])


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_paloc_json(data: dict[str, Any], output_path: Path) -> int:
    text = json.dumps(data, ensure_ascii=False, indent=4)
    _atomic_write_text(output_path, text)
    return len(data["entries"])


def export_paloc_json(source_path: Path, output_path: Path) -> int:
    try:
        data = source_path.read_bytes()
    except OSError as error:
        raise PalocFormatError(f"Cannot read PALOC: {error}.") from error
    return write_paloc_json(paloc_to_json_data(data), output_path)
