from dataclasses import dataclass
from typing import Any

from .archive import LocalizationPaloc
from .paloc import PalocFormatError, json_data_to_paloc, paloc_to_json_data


CATALOG_FORMAT_VERSION = 2
CATALOG_LAYOUT = "multi-paloc"


@dataclass(frozen=True)
class CatalogBuild:
    replacements: tuple[tuple[str, bytes], ...]
    entry_count: int
    paloc_count: int
    matched_entries: int
    new_entries: int
    removed_entries: int
    changed_originals: int


def _identity(entry: dict[str, Any], field_name: str) -> tuple[str, str]:
    marker = entry.get("marker")
    key = entry.get("key")
    if not isinstance(marker, str):
        raise PalocFormatError(f"{field_name}.marker must be hexadecimal text.")
    try:
        marker_bytes = bytes.fromhex(marker)
    except ValueError as error:
        raise PalocFormatError(f"{field_name}.marker is not valid hexadecimal.") from error
    if len(marker_bytes) != 8:
        raise PalocFormatError(f"{field_name}.marker must contain exactly 8 bytes.")
    if not isinstance(key, str):
        raise PalocFormatError(f"{field_name}.key must be text.")
    return marker_bytes.hex(), key


def _paloc_documents(
    palocs: tuple[LocalizationPaloc, ...],
) -> tuple[tuple[LocalizationPaloc, dict[str, Any]], ...]:
    documents: list[tuple[LocalizationPaloc, dict[str, Any]]] = []
    seen: dict[tuple[str, str], str] = {}
    for paloc in palocs:
        document = paloc_to_json_data(paloc.data)
        for index, entry in enumerate(document["entries"]):
            identity = _identity(entry, f"{paloc.path}.entries[{index}]")
            previous = seen.get(identity)
            if previous is not None:
                raise PalocFormatError(
                    f"Duplicate marker + key identity in {previous} and {paloc.path}."
                )
            seen[identity] = paloc.path
        documents.append((paloc, document))
    return tuple(documents)


def palocs_to_catalog(
    palocs: tuple[LocalizationPaloc, ...],
    package_folder: str,
) -> dict[str, Any]:
    documents = _paloc_documents(palocs)
    entries: list[dict[str, str]] = []
    manifest: list[dict[str, Any]] = []
    for paloc, document in documents:
        metadata = document["metadata"]
        manifest.append(
            {
                "path": paloc.path,
                "entry_count": len(document["entries"]),
                "tail_bytes": metadata["tail_bytes"],
            }
        )
        entries.extend(dict(entry) for entry in document["entries"])
    return {
        "metadata": {
            "format_version": CATALOG_FORMAT_VERSION,
            "layout": CATALOG_LAYOUT,
            "package_folder": package_folder,
            "palocs": manifest,
        },
        "entries": entries,
    }


def _validate_catalog_metadata(data: dict[str, Any], package_folder: str) -> None:
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise PalocFormatError("metadata must be an object.")
    if "format_version" not in metadata:
        return
    if metadata.get("format_version") != CATALOG_FORMAT_VERSION:
        raise PalocFormatError("Unsupported localization catalog format_version.")
    if metadata.get("layout") != CATALOG_LAYOUT:
        raise PalocFormatError("Localization catalog layout must be multi-paloc.")
    if metadata.get("package_folder") != package_folder:
        raise PalocFormatError(
            f"Localization catalog targets package {metadata.get('package_folder')!r}, "
            f"not {package_folder}."
        )
    manifest = metadata.get("palocs")
    if not isinstance(manifest, list) or not manifest:
        raise PalocFormatError("Multi-PALOC metadata must contain a palocs manifest.")
    for index, item in enumerate(manifest):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise PalocFormatError(f"metadata.palocs[{index}] must contain a path.")


def build_catalog_replacements(
    data: dict[str, Any],
    source_palocs: tuple[LocalizationPaloc, ...],
    package_folder: str,
) -> CatalogBuild:
    _validate_catalog_metadata(data, package_folder)
    input_entries = data.get("entries")
    if not isinstance(input_entries, list) or not input_entries:
        raise PalocFormatError("JSON must contain a non-empty entries list.")

    translations: dict[tuple[str, str], tuple[str, str]] = {}
    for index, raw_entry in enumerate(input_entries):
        if not isinstance(raw_entry, dict):
            raise PalocFormatError(f"entries[{index}] must be an object.")
        identity = _identity(raw_entry, f"entries[{index}]")
        original = raw_entry.get("original")
        translation = raw_entry.get("translation")
        if not isinstance(original, str):
            raise PalocFormatError(f"entries[{index}].original must be text.")
        if not isinstance(translation, str):
            raise PalocFormatError(f"entries[{index}].translation must be text.")
        if identity in translations:
            raise PalocFormatError(
                f"Duplicate marker + key identity in entries[{index}]."
            )
        translations[identity] = (original, translation)

    documents = _paloc_documents(source_palocs)
    source_identities: set[tuple[str, str]] = set()
    replacements: list[tuple[str, bytes]] = []
    matched_entries = 0
    changed_originals = 0

    for paloc, document in documents:
        rebuilt_entries: list[dict[str, str]] = []
        for index, source_entry in enumerate(document["entries"]):
            identity = _identity(source_entry, f"{paloc.path}.entries[{index}]")
            source_identities.add(identity)
            input_value = translations.get(identity)
            if input_value is None:
                translation = source_entry["original"]
            else:
                input_original, translation = input_value
                matched_entries += 1
                if input_original != source_entry["original"]:
                    changed_originals += 1
            rebuilt_entries.append(
                {
                    "marker": source_entry["marker"],
                    "key": source_entry["key"],
                    "original": source_entry["original"],
                    "translation": translation,
                }
            )
        replacements.append(
            (
                paloc.path,
                json_data_to_paloc(
                    {
                        "metadata": dict(document["metadata"]),
                        "entries": rebuilt_entries,
                    }
                ),
            )
        )

    input_identities = set(translations)
    return CatalogBuild(
        replacements=tuple(replacements),
        entry_count=len(source_identities),
        paloc_count=len(source_palocs),
        matched_entries=matched_entries,
        new_entries=len(source_identities - input_identities),
        removed_entries=len(input_identities - source_identities),
        changed_originals=changed_originals,
    )
