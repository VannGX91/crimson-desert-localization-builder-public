import struct
import unittest

from crimson0020.checksums import pa_checksum
from crimson0020.metadata import MetadataFormatError, parse_papgt, patch_folder_crc


def build_papgt(entries: list[tuple[str, int]]) -> bytes:
    strings = bytearray()
    records = bytearray()
    for index, (name, crc) in enumerate(entries):
        string_offset = len(strings)
        strings += name.encode("utf-8") + b"\x00"
        records += struct.pack("<III", 0x1000 + index, string_offset, crc)

    data = bytearray(b"PAPG\x00\x00\x00\x00")
    data += bytes((len(entries), 0xA1, 0xB2, 0xC3))
    data += records
    data += struct.pack("<I", len(strings))
    data += strings
    struct.pack_into("<I", data, 4, pa_checksum(data[12:]))
    return bytes(data)


class MetadataTests(unittest.TestCase):
    def test_parse_maps_folder_names_and_crc_offsets(self) -> None:
        data = build_papgt(
            [("0019", 0x11111111), ("0020", 0x22222222), ("0021", 0x33333333)]
        )

        index = parse_papgt(data)

        self.assertEqual(3, len(index.entries))
        self.assertEqual(0x22222222, index.entry_for("0020").crc)
        self.assertEqual(32, index.entry_for("0020").crc_offset)
        self.assertEqual(index.stored_checksum, index.computed_checksum)

    def test_patch_changes_only_target_crc_and_self_checksum(self) -> None:
        original = build_papgt(
            [("0019", 0x11111111), ("0020", 0x22222222), ("0021", 0x33333333)]
        )
        target_offset = parse_papgt(original).entry_for("0020").crc_offset

        patched = patch_folder_crc(original, "0020", 0xDEADBEEF)

        parsed = parse_papgt(patched)
        self.assertEqual(0xDEADBEEF, parsed.entry_for("0020").crc)
        self.assertEqual(pa_checksum(patched[12:]), struct.unpack_from("<I", patched, 4)[0])

        original_without_allowed_fields = bytearray(original)
        patched_without_allowed_fields = bytearray(patched)
        for offset in (*range(4, 8), *range(target_offset, target_offset + 4)):
            original_without_allowed_fields[offset] = 0
            patched_without_allowed_fields[offset] = 0
        self.assertEqual(original_without_allowed_fields, patched_without_allowed_fields)

    def test_malformed_or_bad_checksum_metadata_is_rejected(self) -> None:
        valid = build_papgt([("0020", 0x22222222)])

        with self.assertRaises(MetadataFormatError):
            parse_papgt(valid[:-1])

        damaged = bytearray(valid)
        damaged[-1] ^= 0x01
        with self.assertRaisesRegex(MetadataFormatError, "checksum"):
            parse_papgt(damaged)

    def test_patch_rejects_metadata_without_0020(self) -> None:
        data = build_papgt([("0019", 0x11111111)])

        with self.assertRaisesRegex(MetadataFormatError, "0020"):
            patch_folder_crc(data, "0020", 0x22222222)


if __name__ == "__main__":
    unittest.main()
