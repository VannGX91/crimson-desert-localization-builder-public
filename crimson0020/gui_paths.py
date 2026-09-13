import os
import sys
from pathlib import Path

from .languages import LanguageSpec


LAUNCHER_DIRECTORY_ENV = "CRIMSON_LAUNCHER_DIRECTORY"


def application_directory() -> Path:
    launcher_directory = os.environ.get(LAUNCHER_DIRECTORY_ENV, "").strip()
    if launcher_directory:
        return Path(launcher_directory).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resolve_portable_path(
    path_value: str | Path,
    base_directory: str | Path,
) -> Path:
    value = str(path_value).strip()
    if not value:
        raise ValueError("Please select a path.")
    base = Path(base_directory).expanduser().resolve()
    selected = Path(value).expanduser()
    if not selected.is_absolute():
        selected = base / selected
    return selected.resolve()


def portable_path_text(
    path_value: str | Path,
    base_directory: str | Path,
) -> str:
    base = Path(base_directory).expanduser().resolve()
    resolved = resolve_portable_path(path_value, base)
    try:
        relative = resolved.relative_to(base)
    except ValueError:
        return str(resolved)
    return str(relative) if relative.parts else "."


def default_localization_source(spec: LanguageSpec) -> str:
    steam_package = Path(
        r"C:\Program Files (x86)\Steam\steamapps\common\Crimson Desert"
    ) / spec.package_folder
    return str(steam_package) if steam_package.is_dir() else ""


def default_font_source() -> str:
    steam_0012 = Path(
        r"C:\Program Files (x86)\Steam\steamapps\common\Crimson Desert\0012"
    )
    return str(steam_0012) if steam_0012.is_dir() else ""


def default_game() -> str:
    steam_game = Path(
        r"C:\Program Files (x86)\Steam\steamapps\common\Crimson Desert"
    )
    return str(steam_game) if (steam_game / "meta" / "0.papgt").is_file() else ""


def unpack_json_output_path(
    output_folder: str | Path,
    json_filename: str = "localizationstring_eng.paloc.json",
) -> Path:
    value = str(output_folder).strip()
    if not value:
        raise ValueError("Please select an output folder.")
    return Path(value).expanduser().resolve() / json_filename


def _named_child(path_value: str | Path, folder_name: str, message: str) -> Path:
    value = str(path_value).strip()
    if not value:
        raise ValueError(message)
    selected = Path(value).expanduser().resolve()
    return selected if selected.name.casefold() == folder_name else selected / folder_name


def font_unpack_output_path(output_location: str | Path) -> Path:
    return _named_child(
        output_location,
        "fonts",
        "Please select the output location for unpacked fonts.",
    )


def font_package_output_path(output_location: str | Path) -> Path:
    return _named_child(
        output_location,
        "0012",
        "Please select the output location for package 0012.",
    )


def default_output_directories(base_directory: str | Path) -> tuple[Path, Path]:
    base = Path(base_directory).expanduser().resolve()
    return base / "output_unpack", base / "output_repack"


def default_install_source_directory(base_directory: str | Path) -> Path:
    base = Path(base_directory).expanduser().resolve()
    return base / "output_repack"


def ensure_named_output_directory(selected_path: str | Path, folder_name: str) -> Path:
    if folder_name not in {"output_unpack", "output_repack"}:
        raise ValueError(f"Unsupported output folder name: {folder_name}")
    output = _named_child(
        selected_path,
        folder_name,
        "Please select an output location.",
    )
    output.mkdir(parents=True, exist_ok=True)
    return output
