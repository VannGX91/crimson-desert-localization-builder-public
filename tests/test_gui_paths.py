import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import crimson0020.gui_paths as gui_paths
from crimson0020.gui_paths import (
    default_install_source_directory,
    default_output_directories,
    ensure_named_output_directory,
    font_package_output_path,
    font_unpack_output_path,
    portable_path_text,
    resolve_portable_path,
    unpack_json_output_path,
)


class GuiPathTests(unittest.TestCase):
    def test_native_launcher_directory_controls_one_folder_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            launcher_directory = Path(temp_dir) / "public_release"

            with patch.dict(
                gui_paths.os.environ,
                {"CRIMSON_LAUNCHER_DIRECTORY": str(launcher_directory)},
            ):
                result = gui_paths.application_directory()

            self.assertEqual(launcher_directory.resolve(), result)

    def test_frozen_app_uses_executable_directory_for_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            executable_directory = Path(temp_dir) / "public_release"
            executable = executable_directory / "CrimsonDesert0020Tool.exe"

            with (
                patch.object(gui_paths.sys, "frozen", True, create=True),
                patch.object(gui_paths.sys, "executable", str(executable)),
            ):
                result = gui_paths.application_directory()

            self.assertEqual(executable_directory.resolve(), result)

    def test_default_outputs_are_separate_unpack_and_repack_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_directory = Path(temp_dir)

            unpack_output, repack_output = default_output_directories(base_directory)

            self.assertEqual(base_directory.resolve() / "output_unpack", unpack_output)
            self.assertEqual(base_directory.resolve() / "output_repack", repack_output)

    def test_selecting_parent_creates_named_unpack_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_parent = Path(temp_dir) / "public_build"

            output = ensure_named_output_directory(selected_parent, "output_unpack")

            self.assertEqual(selected_parent.resolve() / "output_unpack", output)
            self.assertTrue(output.is_dir())

    def test_selecting_named_folder_does_not_nest_it_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_output = Path(temp_dir) / "output_repack"
            selected_output.mkdir()

            output = ensure_named_output_directory(selected_output, "output_repack")

            self.assertEqual(selected_output.resolve(), output)
            self.assertFalse((selected_output / "output_repack").exists())

    def test_unpack_json_is_created_inside_selected_output_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_output = Path(temp_dir) / "output_test"

            result = unpack_json_output_path(str(selected_output))

            self.assertEqual(
                selected_output.resolve() / "localizationstring_eng.paloc.json",
                result,
            )

    def test_unpack_requires_an_output_folder(self) -> None:
        with self.assertRaisesRegex(ValueError, "output"):
            unpack_json_output_path("   ")

    def test_install_defaults_to_repack_root_for_language_and_font_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_directory = Path(temp_dir)

            result = default_install_source_directory(base_directory)

            self.assertEqual(
                base_directory.resolve() / "output_repack",
                result,
            )

    def test_font_repack_output_names_the_resulting_0012_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / "custom_font_output"
            exact = parent / "0012"

            self.assertEqual(exact.resolve(), font_package_output_path(parent))
            self.assertEqual(exact.resolve(), font_package_output_path(exact))

    def test_font_unpack_output_uses_a_dedicated_fonts_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / "output_test"
            exact = parent / "fonts"

            self.assertEqual(exact.resolve(), font_unpack_output_path(parent))
            self.assertEqual(exact.resolve(), font_unpack_output_path(exact))

    def test_paths_below_application_folder_are_displayed_portably(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_directory = Path(temp_dir)
            json_path = (
                base_directory
                / "output_unpack"
                / "localizationstring_eng.paloc.json"
            )

            displayed = portable_path_text(json_path, base_directory)
            resolved = resolve_portable_path(displayed, base_directory)

            self.assertEqual(
                str(Path("output_unpack") / "localizationstring_eng.paloc.json"),
                displayed,
            )
            self.assertEqual(json_path.resolve(), resolved)

    def test_custom_absolute_path_stays_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_directory = Path(temp_dir) / "app"
            custom_path = Path(temp_dir) / "custom" / "output_repack"

            displayed = portable_path_text(custom_path, base_directory)
            resolved = resolve_portable_path(displayed, base_directory)

            self.assertEqual(str(custom_path.resolve()), displayed)
            self.assertEqual(custom_path.resolve(), resolved)


if __name__ == "__main__":
    unittest.main()
