import struct
from pathlib import Path

from crimson0020.checksums import pa_checksum
from crimson0020.crypto import compress_lz4, encrypt
from crimson0020.paloc import json_data_to_paloc


def pack_named_node(parent: int, name: str) -> bytes:
    encoded = name.encode("utf-8")
    return struct.pack("<IB", parent, len(encoded)) + encoded


def valid_ttf(label: str) -> bytes:
    return b"\x00\x01\x00\x00" + label.encode("utf-8") * 32


def build_localization_package(
    root: Path,
    package_folder: str,
    paloc_filename: str,
    translation: str = "Hello",
) -> Path:
    package = root / package_folder
    package.mkdir(parents=True)
    plaintext = json_data_to_paloc(
        {
            "metadata": {"tail_bytes": ""},
            "entries": [
                {
                    "marker": "0102030405060708",
                    "key": "UI_HELLO",
                    "original": translation,
                    "translation": translation,
                }
            ],
        }
    )
    payload = encrypt(compress_lz4(plaintext), paloc_filename)
    paz_data = payload + b"\x00" * ((16 - len(payload) % 16) % 16)
    (package / "0.paz").write_bytes(paz_data)

    folder_section = pack_named_node(0xFFFFFFFF, "gamedata")
    node_section = pack_named_node(0xFFFFFFFF, paloc_filename)
    pamt = bytearray(b"\x00\x00\x00\x00")
    pamt += struct.pack("<I", 1)
    pamt += struct.pack("<II", 0, 0)
    pamt += struct.pack("<II", pa_checksum(paz_data), len(paz_data))
    pamt += struct.pack("<I", len(folder_section)) + folder_section
    pamt += struct.pack("<I", len(node_section)) + node_section
    pamt += struct.pack("<II", 0, 0)
    pamt += struct.pack(
        "<IIIII",
        0,
        0,
        len(payload),
        len(plaintext),
        0x00020000,
    )
    struct.pack_into("<I", pamt, 0, pa_checksum(pamt[12:]))
    (package / "0.pamt").write_bytes(pamt)
    return package


def build_multi_paloc_package(
    root: Path,
    package_folder: str = "0020",
    entries_by_paloc: dict[str, list[dict[str, str]]] | None = None,
) -> Path:
    package = root / package_folder
    package.mkdir(parents=True)
    palocs = entries_by_paloc or {
        "aidialog.paloc": [
            {
                "marker": "0101010101010101",
                "key": "DIALOG_HELLO",
                "original": "Hello",
                "translation": "Hello",
            }
        ],
        "quest.paloc": [
            {
                "marker": "0202020202020202",
                "key": "QUEST_START",
                "original": "Begin the quest",
                "translation": "Begin the quest",
            }
        ],
        "ui.paloc": [
            {
                "marker": "0303030303030303",
                "key": "UI_CONFIRM",
                "original": "Confirm",
                "translation": "Confirm",
            }
        ],
    }

    paz_data = bytearray()
    entry_records: list[tuple[str, int, int, int, int]] = []
    for index, (filename, entries) in enumerate(palocs.items(), start=1):
        plaintext = json_data_to_paloc(
            {
                "metadata": {"tail_bytes": f"{index:08x}"},
                "entries": entries,
            }
        )
        payload = encrypt(compress_lz4(plaintext), filename)
        aligned = (len(paz_data) + 15) & ~15
        paz_data += b"\x00" * (aligned - len(paz_data))
        offset = len(paz_data)
        paz_data += payload
        entry_records.append(
            (filename, offset, len(payload), len(plaintext), 0x00020000)
        )
    paz_data += b"\x00" * ((16 - len(paz_data) % 16) % 16)
    (package / "0.paz").write_bytes(paz_data)

    folder_section = pack_named_node(0xFFFFFFFF, "gamedata")
    node_section = bytearray()
    node_offsets: dict[str, int] = {}
    for filename in palocs:
        node_offsets[filename] = len(node_section)
        node_section += pack_named_node(0xFFFFFFFF, filename)

    pamt = bytearray(b"\x00\x00\x00\x00")
    pamt += struct.pack("<I", 1)
    pamt += struct.pack("<II", 0, 0)
    pamt += struct.pack("<II", pa_checksum(paz_data), len(paz_data))
    pamt += struct.pack("<I", len(folder_section)) + folder_section
    pamt += struct.pack("<I", len(node_section)) + node_section
    pamt += struct.pack("<II", 0, 0)
    for filename, offset, comp_size, orig_size, flags in entry_records:
        pamt += struct.pack(
            "<IIIII",
            node_offsets[filename],
            offset,
            comp_size,
            orig_size,
            flags,
        )
    struct.pack_into("<I", pamt, 0, pa_checksum(pamt[12:]))
    (package / "0.pamt").write_bytes(pamt)
    return package


def build_font_package(
    root: Path,
    language_font: str,
    label: str = "original",
) -> tuple[Path, dict[str, bytes]]:
    package = root / "0012"
    package.mkdir(parents=True)
    font_data = {
        "basefont.ttf": valid_ttf(f"base-{label}"),
        language_font: valid_ttf(f"language-{label}"),
        "creditfont.ttf": valid_ttf(f"credit-{label}"),
    }
    unrelated = b"unrelated-payload"

    paz_zero = bytearray()
    entry_records: list[tuple[str, int, int, int, int]] = []
    for filename, plaintext in font_data.items():
        aligned = (len(paz_zero) + 15) & ~15
        paz_zero += b"\x00" * (aligned - len(paz_zero))
        payload = compress_lz4(plaintext)
        offset = len(paz_zero)
        paz_zero += payload
        entry_records.append(
            (filename, offset, len(payload), len(plaintext), 0x00020000)
        )
    paz_zero += b"\x00" * ((16 - len(paz_zero) % 16) % 16)
    paz_one = unrelated + b"\x00" * ((16 - len(unrelated) % 16) % 16)
    (package / "0.paz").write_bytes(paz_zero)
    (package / "1.paz").write_bytes(paz_one)

    folder_section = pack_named_node(0xFFFFFFFF, "ui")
    node_section = bytearray()
    node_offsets: dict[str, int] = {}
    for filename in (*font_data.keys(), "unrelated.bin"):
        node_offsets[filename] = len(node_section)
        node_section += pack_named_node(0xFFFFFFFF, filename)

    pamt = bytearray(b"\x00\x00\x00\x00")
    pamt += struct.pack("<I", 2)
    pamt += struct.pack("<II", 0, 0)
    pamt += struct.pack("<II", pa_checksum(paz_zero), len(paz_zero))
    pamt += struct.pack("<I", 0)
    pamt += struct.pack("<II", pa_checksum(paz_one), len(paz_one))
    pamt += struct.pack("<I", len(folder_section)) + folder_section
    pamt += struct.pack("<I", len(node_section)) + node_section
    pamt += struct.pack("<II", 0, 0)
    for filename, offset, comp_size, orig_size, flags in entry_records:
        pamt += struct.pack(
            "<IIIII",
            node_offsets[filename],
            offset,
            comp_size,
            orig_size,
            flags,
        )
    pamt += struct.pack(
        "<IIIII",
        node_offsets["unrelated.bin"],
        0,
        len(unrelated),
        len(unrelated),
        0x00000001,
    )
    struct.pack_into("<I", pamt, 0, pa_checksum(pamt[12:]))
    (package / "0.pamt").write_bytes(pamt)
    return package, font_data
