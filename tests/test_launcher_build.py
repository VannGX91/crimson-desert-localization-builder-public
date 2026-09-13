import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from build_launcher import launcher_compile_command
from scripts.build_onedir import (
    APP_FOLDER_NAME,
    ARCHIVE_NAME,
    INTERNAL_FOLDER_NAME,
    LAUNCHER_EXE_NAME,
    validate_candidate_layout,
    write_build_info,
)
from scripts.build_portable import create_deterministic_zip, write_checksums


class LauncherBuildTests(unittest.TestCase):
    def test_release_names_identify_a_one_folder_windows_build(self) -> None:
        self.assertEqual("CrimsonDesertLocalizationBuilder", APP_FOLDER_NAME)
        self.assertEqual("_internal", INTERNAL_FOLDER_NAME)
        self.assertEqual(
            "CrimsonDesertLocalizationBuilder-windows-x64.zip",
            ARCHIVE_NAME,
        )
        self.assertEqual(
            "Crimson.Desert.Localization.Builder.exe",
            LAUNCHER_EXE_NAME,
        )

    def test_launcher_source_starts_only_the_internal_application(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        source = (project_root / "launcher" / "ToolLauncher.cs").read_text(
            encoding="utf-8"
        )

        for required_text in (
            '"_internal", "runtime", "pythonw.exe"',
            '"_internal", "app", "app.py"',
            "ProcessStartInfo",
            "UseShellExecute = false",
            'EnvironmentVariables["CRIMSON_LAUNCHER_DIRECTORY"]',
            '"--smoke-test"',
            "giải nén lại toàn bộ file ZIP",
            "child.ExitCode != 0",
            "Microsoft Edge WebView2 Runtime",
        ):
            self.assertIn(required_text, source)
        for forbidden_text in (
            "cmd.exe",
            "ShellExecute(",
            "system(",
            "CACHED_PAYLOAD",
            "CachedPayload",
            "GetManifestResourceStream",
            "ZipArchive",
            "TCL_LIBRARY",
            "TK_LIBRARY",
        ):
            self.assertNotIn(forbidden_text, source)

    def test_build_info_declares_pre_extracted_one_folder_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            internal = Path(temp_dir) / INTERNAL_FOLDER_NAME
            internal.mkdir()

            build_info_path = write_build_info(
                internal,
                ("cryptography==50.0.0", "lz4==4.4.5"),
            )
            build_info = json.loads(build_info_path.read_text(encoding="utf-8"))

            self.assertEqual("pre-extracted-one-folder", build_info["layout"])
            self.assertFalse(build_info["first_run_extraction"])
            self.assertFalse(build_info["embedded_payload"])
            self.assertEqual("_internal", build_info["internal_folder"])
            self.assertEqual(
                "crimson0020/assets/crimson-desert-localization.ico",
                build_info["application_icon"],
            )
            self.assertTrue(build_info["launcher_icon_embedded"])

    def test_zip_opens_to_one_folder_with_exe_and_internal_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = root / APP_FOLDER_NAME
            internal = candidate / INTERNAL_FOLDER_NAME
            internal.mkdir(parents=True)
            (candidate / LAUNCHER_EXE_NAME).write_bytes(b"launcher")
            (internal / "runtime.dat").write_bytes(b"runtime")
            destination = root / ARCHIVE_NAME

            create_deterministic_zip(candidate, destination)

            with zipfile.ZipFile(destination) as archive:
                names = archive.namelist()
            prefix = f"{APP_FOLDER_NAME}/"
            self.assertTrue(all(name.startswith(prefix) for name in names))
            root_entries = {
                name.removeprefix(prefix).split("/", 1)[0] for name in names
            }
            self.assertEqual(
                {LAUNCHER_EXE_NAME, INTERNAL_FOLDER_NAME},
                root_entries,
            )

    def test_launcher_build_uses_deterministic_microsoft_compiler(self) -> None:
        command = launcher_compile_command(
            dotnet=Path("dotnet.exe"),
            compiler=Path("csc.dll"),
            framework=Path("framework"),
            source=Path("ToolLauncher.cs"),
            destination=Path("Tool.exe"),
            icon=Path("crimson-desert-localization.ico"),
        )

        self.assertEqual(("dotnet.exe", "csc.dll"), command[:2])
        self.assertIn("/target:winexe", command)
        self.assertIn("/platform:x64", command)
        self.assertIn("/deterministic+", command)
        self.assertIn(
            "/win32icon:crimson-desert-localization.ico",
            command,
        )
        self.assertIn("/nostdlib+", command)
        self.assertIn("/reference:framework\\mscorlib.dll", command)
        self.assertIn("/reference:framework\\System.dll", command)
        self.assertIn(
            "/reference:framework\\System.Windows.Forms.dll",
            command,
        )
        self.assertNotIn("ziglang", " ".join(command).casefold())

    def test_candidate_root_contains_only_launcher_and_internal_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / APP_FOLDER_NAME
            internal = candidate / INTERNAL_FOLDER_NAME
            application = internal / "app"
            runtime = internal / "runtime"
            native = application / "crimson0020" / "native"
            assets = application / "crimson0020" / "assets"
            native.mkdir(parents=True)
            assets.mkdir(parents=True)
            runtime.mkdir(parents=True)
            (candidate / LAUNCHER_EXE_NAME).write_bytes(b"launcher")
            for filename in (
                "pythonw.exe",
                "python3.dll",
                "python314.dll",
                "vcruntime140.dll",
                "vcruntime140_1.dll",
            ):
                (runtime / filename).write_bytes(b"runtime")
            (runtime / "python314._pth").write_text(
                "..\\app\n", encoding="utf-8"
            )
            (application / "app.py").write_text("pass\n", encoding="utf-8")
            (native / "pa_checksum.dll").write_bytes(b"helper")
            (assets / "crimson-desert-localization.ico").write_bytes(b"icon")
            (assets / "crimson-desert-localization.png").write_bytes(b"preview")
            (internal / "BUILD_INFO.json").write_text("{}\n", encoding="utf-8")
            write_checksums(candidate, internal / "SHA256SUMS.txt")

            validate_candidate_layout(candidate)
            self.assertEqual(
                {LAUNCHER_EXE_NAME, INTERNAL_FOLDER_NAME},
                {path.name for path in candidate.iterdir()},
            )
            manifest = (internal / "SHA256SUMS.txt").read_text(encoding="ascii")
            self.assertIn(LAUNCHER_EXE_NAME, manifest)
            self.assertNotIn("SHA256SUMS.txt", manifest)

            icon = assets / "crimson-desert-localization.ico"
            icon.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "localization.ico"):
                validate_candidate_layout(candidate)
            icon.write_bytes(b"icon")

            (candidate / "unexpected.dll").write_bytes(b"wrong root file")
            with self.assertRaisesRegex(ValueError, "exactly the launcher"):
                validate_candidate_layout(candidate)

    def test_candidate_layout_rejects_missing_internal_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / APP_FOLDER_NAME
            internal = candidate / INTERNAL_FOLDER_NAME
            application = internal / "app"
            native = application / "crimson0020" / "native"
            native.mkdir(parents=True)
            (candidate / LAUNCHER_EXE_NAME).write_bytes(b"launcher")
            (application / "app.py").write_text("pass\n", encoding="utf-8")
            (native / "pa_checksum.dll").write_bytes(b"helper")
            (internal / "BUILD_INFO.json").write_text("{}\n", encoding="utf-8")
            (internal / "SHA256SUMS.txt").write_text("manifest\n", encoding="ascii")

            with self.assertRaisesRegex(FileNotFoundError, "pythonw.exe"):
                validate_candidate_layout(candidate)


if __name__ == "__main__":
    unittest.main()
