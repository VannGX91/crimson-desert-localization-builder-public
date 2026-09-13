import os
import struct

import lz4.block
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms


HASH_INITIAL_VALUE = 0x000C5EDE
IV_XOR = 0x60616263
XOR_DELTAS = (
    0x00000000,
    0x0A0A0A0A,
    0x0C0C0C0C,
    0x06060606,
    0x0E0E0E0E,
    0x0A0A0A0A,
    0x06060606,
    0x02020202,
)
UINT32_MASK = 0xFFFFFFFF


def _rotate(value: int, count: int) -> int:
    return ((value << count) | (value >> (32 - count))) & UINT32_MASK


def _add(left: int, right: int) -> int:
    return (left + right) & UINT32_MASK


def _subtract(left: int, right: int) -> int:
    return (left - right) & UINT32_MASK


def hashlittle(data: bytes, initial_value: int = 0) -> int:
    """Return Bob Jenkins' lookup3 hashlittle primary hash."""
    remaining = len(data)
    a = b = c = _add(0xDEADBEEF + remaining, initial_value)
    offset = 0

    while remaining > 12:
        a = _add(a, struct.unpack_from("<I", data, offset)[0])
        b = _add(b, struct.unpack_from("<I", data, offset + 4)[0])
        c = _add(c, struct.unpack_from("<I", data, offset + 8)[0])
        a = _subtract(a, c)
        a ^= _rotate(c, 4)
        c = _add(c, b)
        b = _subtract(b, a)
        b ^= _rotate(a, 6)
        a = _add(a, c)
        c = _subtract(c, b)
        c ^= _rotate(b, 8)
        b = _add(b, a)
        a = _subtract(a, c)
        a ^= _rotate(c, 16)
        c = _add(c, b)
        b = _subtract(b, a)
        b ^= _rotate(a, 19)
        a = _add(a, c)
        c = _subtract(c, b)
        c ^= _rotate(b, 4)
        b = _add(b, a)
        offset += 12
        remaining -= 12

    tail = data[offset:] + (b"\x00" * 12)
    if remaining >= 12:
        c = _add(c, struct.unpack_from("<I", tail, 8)[0])
    elif remaining >= 9:
        value = struct.unpack_from("<I", tail, 8)[0]
        c = _add(c, value & (UINT32_MASK >> (8 * (12 - remaining))))
    if remaining >= 8:
        b = _add(b, struct.unpack_from("<I", tail, 4)[0])
    elif remaining >= 5:
        value = struct.unpack_from("<I", tail, 4)[0]
        b = _add(b, value & (UINT32_MASK >> (8 * (8 - remaining))))
    if remaining >= 4:
        a = _add(a, struct.unpack_from("<I", tail, 0)[0])
    elif remaining >= 1:
        value = struct.unpack_from("<I", tail, 0)[0]
        a = _add(a, value & (UINT32_MASK >> (8 * (4 - remaining))))
    else:
        return c

    c ^= b
    c = _subtract(c, _rotate(b, 14))
    a ^= c
    a = _subtract(a, _rotate(c, 11))
    b ^= a
    b = _subtract(b, _rotate(a, 25))
    c ^= b
    c = _subtract(c, _rotate(b, 16))
    a ^= c
    a = _subtract(a, _rotate(c, 4))
    b ^= a
    b = _subtract(b, _rotate(a, 14))
    c ^= b
    return _subtract(c, _rotate(b, 24))


def derive_key_and_iv(filename: str) -> tuple[bytes, bytes]:
    basename = os.path.basename(filename).lower()
    seed = hashlittle(basename.encode("utf-8"), HASH_INITIAL_VALUE)
    iv = struct.pack("<I", seed) * 4
    key_base = seed ^ IV_XOR
    key = b"".join(struct.pack("<I", key_base ^ delta) for delta in XOR_DELTAS)
    return key, iv


def _chacha20(data: bytes, filename: str) -> bytes:
    key, iv = derive_key_and_iv(filename)
    cipher = Cipher(algorithms.ChaCha20(key, iv), mode=None)
    return cipher.encryptor().update(data)


def encrypt(data: bytes, filename: str) -> bytes:
    return _chacha20(data, filename)


def decrypt(data: bytes, filename: str) -> bytes:
    return _chacha20(data, filename)


def compress_lz4(data: bytes) -> bytes:
    return lz4.block.compress(data, store_size=False)


def decompress_lz4(data: bytes, original_size: int) -> bytes:
    return lz4.block.decompress(data, uncompressed_size=original_size)
