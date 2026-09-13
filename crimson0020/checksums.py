import struct
from pathlib import Path
from typing import BinaryIO

from .native_checksum import native_pa_checksum_file


PA_MAGIC = 0x2145E233
UINT32_MASK = 0xFFFFFFFF


def _rotate_left(value: int, count: int) -> int:
    return ((value << count) & UINT32_MASK) | (value >> (32 - count))


def _rotate_right(value: int, count: int) -> int:
    return (value >> count) | ((value << (32 - count)) & UINT32_MASK)


def _mix_values(
    a: int,
    b: int,
    c: int,
    first: int,
    second: int,
    third: int,
) -> tuple[int, int, int]:
    a = (a + first) & UINT32_MASK
    b = (b + second) & UINT32_MASK
    c = (c + third) & UINT32_MASK

    a = (a - c) & UINT32_MASK
    a ^= _rotate_left(c, 4)
    c = (c + b) & UINT32_MASK
    b = (b - a) & UINT32_MASK
    b ^= _rotate_left(a, 6)
    a = (a + c) & UINT32_MASK
    c = (c - b) & UINT32_MASK
    c ^= _rotate_left(b, 8)
    b = (b + a) & UINT32_MASK
    a = (a - c) & UINT32_MASK
    a ^= _rotate_left(c, 16)
    c = (c + b) & UINT32_MASK
    b = (b - a) & UINT32_MASK
    b ^= _rotate_left(a, 19)
    a = (a + c) & UINT32_MASK
    c = (c - b) & UINT32_MASK
    c ^= _rotate_left(b, 4)
    b = (b + a) & UINT32_MASK
    return a, b, c


def _mix_block(a: int, b: int, c: int, block: bytes) -> tuple[int, int, int]:
    return _mix_values(a, b, c, *struct.unpack_from("<III", block))


def _finish_checksum(a: int, b: int, c: int, tail: bytes) -> int:
    for index, byte in enumerate(tail):
        if index < 4:
            a = (a + (byte << (8 * index))) & UINT32_MASK
        elif index < 8:
            b = (b + (byte << (8 * (index - 4)))) & UINT32_MASK
        else:
            c = (c + (byte << (8 * (index - 8)))) & UINT32_MASK

    value_82 = ((b ^ c) - _rotate_left(b, 14)) & UINT32_MASK
    value_83 = ((a ^ value_82) - _rotate_left(value_82, 11)) & UINT32_MASK
    value_84 = ((value_83 ^ b) - _rotate_right(value_83, 7)) & UINT32_MASK
    value_85 = ((value_84 ^ value_82) - _rotate_left(value_84, 16)) & UINT32_MASK
    mixed = ((value_83 ^ value_85) - _rotate_left(value_85, 4)) & UINT32_MASK
    value_87 = ((mixed ^ value_84) - _rotate_left(mixed, 14)) & UINT32_MASK
    return ((value_87 ^ value_85) - _rotate_right(value_87, 8)) & UINT32_MASK


def pa_checksum(data: bytes | bytearray) -> int:
    """Return the 32-bit Pearl Abyss checksum used by PAZ metadata."""
    length = len(data)
    if length == 0:
        return 0

    a = b = c = (length - PA_MAGIC) & UINT32_MASK
    offset = 0
    remaining = length

    while remaining > 12:
        a, b, c = _mix_block(a, b, c, bytes(data[offset : offset + 12]))
        offset += 12
        remaining -= 12

    return _finish_checksum(a, b, c, bytes(data[offset:]))


def _checksum_stream(stream: BinaryIO, length: int) -> int:
    if length == 0:
        return 0
    a = b = c = (length - PA_MAGIC) & UINT32_MASK
    block_count = (length - 1) // 12
    processed = 0
    blocks_per_chunk = 65_536
    while block_count:
        current_blocks = min(block_count, blocks_per_chunk)
        chunk_size = current_blocks * 12
        chunk = stream.read(chunk_size)
        if len(chunk) != chunk_size:
            raise OSError("File ended while calculating its PA checksum.")
        for first, second, third in struct.iter_unpack("<III", chunk):
            a, b, c = _mix_values(a, b, c, first, second, third)
        processed += chunk_size
        block_count -= current_blocks
    remaining = length - processed
    tail = stream.read(remaining)
    if len(tail) != remaining:
        raise OSError("File ended while calculating its PA checksum.")
    return _finish_checksum(a, b, c, tail)


def pa_checksum_file(path: Path | str) -> int:
    file_path = Path(path)
    native_checksum = native_pa_checksum_file(file_path)
    if native_checksum is not None:
        return native_checksum
    length = file_path.stat().st_size
    with file_path.open("rb") as stream:
        return _checksum_stream(stream, length)
