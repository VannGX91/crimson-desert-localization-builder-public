import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from build_native import compiler_command
from scripts.build_portable import (
    APP_FOLDER_NAME,
    OFFICIAL_RUNTIME_FILENAME,
    OFFICIAL_RUNTIME_SHA256,
    OFFICIAL_RUNTIME_URL,
    copy_application_sources,
    create_deterministic_zip,
    prune_runtime,
    remove_generated_bytecode,
    safe_extract_zip,
    verify_runtime_archive,
    write_isolated_path_file,
    write_launcher,
)


class PortableBuildTests(unittest.TestCase):
    def test_application_copy_includes_web_ui_and_vendored_proxy_source(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "app"

            copy_application_sources(project_root, destination)

            for relative in (
                Path("crimson0020/assets/crimson-desert-localization.ico"),
                Path("crimson0020/assets/crimson-desert-localization.png"),
                Path("crimson0020/webui/index.html"),
                Path("crimson0020/webui/styles.css"),
                Path("crimson0020/webui/app.js"),
                Path("cloudscraper/__init__.py"),
                Path("proxy_tools/__init__.py"),
                Path("THIRD_PARTY_NOTICES.md"),
            ):
                self.assertTrue((destination / relative).is_file(), relative)

    def test_official_runtime_is_versioned_and_hash_pinned(self) -> None:
        self.assertEqual("python-3.14.7-amd64.zip", OFFICIAL_RUNTIME_FILENAME)
        self.assertEqual(
            "https://www.python.org/ftp/python/3.14.7/"
            "python-3.14.7-amd64.zip",
            OFFICIAL_RUNTIME_URL,
        )
        self.assertEqual(64, len(OFFICIAL_RUNTIME_SHA256))
        int(OFFICIAL_RUNTIME_SHA256, 16)

    def test_runtime_archive_must_match_the_pinned_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / OFFICIAL_RUNTIME_FILENAME
            archive.write_bytes(b"not the official runtime")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_runtime_archive(archive)

    def test_zip_extraction_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escaped.txt", "unsafe")

            with self.assertRaisesRegex(ValueError, "unsafe path"):
                safe_extract_zip(archive, root / "runtime")

            self.assertFalse((root / "escaped.txt").exists())

    def test_application_copy_uses_an_explicit_source_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            destination = root / "candidate" / "app"
            package = project / "crimson0020"
            tests = project / "tests"
            package.mkdir(parents=True)
            tests.mkdir()
            (project / "app.py").write_text("print('app')\n", encoding="utf-8")
            (project / "sitecustomize.py").write_text(
                "# release bootstrap\n", encoding="utf-8"
            )
            (project / "LICENSE").write_text("license\n", encoding="utf-8")
            (project / "README.md").write_text("readme\n", encoding="utf-8")
            (project / "THIRD_PARTY_NOTICES.md").write_text(
                "notices\n",
                encoding="utf-8",
            )
            cloudscraper = project / "cloudscraper"
            cloudscraper.mkdir()
            (cloudscraper / "__init__.py").write_text(
                "def create_scraper(): raise RuntimeError\n",
                encoding="utf-8",
            )
            proxy_tools = project / "proxy_tools"
            proxy_tools.mkdir()
            (proxy_tools / "__init__.py").write_text(
                "class Proxy: pass\n",
                encoding="utf-8",
            )
            (package / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
            (package / "feature.py").write_text("VALUE = 2\n", encoding="utf-8")
            assets = package / "assets"
            assets.mkdir()
            (assets / "crimson-desert-localization.ico").write_bytes(b"icon")
            (assets / "crimson-desert-localization.png").write_bytes(b"image")
            (package / "private.paz").write_bytes(b"game data")
            (tests / "test_private.py").write_text("secret = 1\n", encoding="utf-8")

            copy_application_sources(project, destination)

            self.assertTrue((destination / "app.py").is_file())
            self.assertTrue((destination / "sitecustomize.py").is_file())
            self.assertTrue((destination / "LICENSE").is_file())
            self.assertTrue((destination / "README.md").is_file())
            self.assertTrue((destination / "THIRD_PARTY_NOTICES.md").is_file())
            self.assertTrue((destination / "cloudscraper" / "__init__.py").is_file())
            self.assertTrue((destination / "proxy_tools" / "__init__.py").is_file())
            self.assertTrue((destination / "crimson0020" / "feature.py").is_file())
            self.assertTrue(
                (
                    destination
                    / "crimson0020"
                    / "assets"
                    / "crimson-desert-localization.ico"
                ).is_file()
            )
            self.assertFalse((destination / "crimson0020" / "private.paz").exists())
            self.assertFalse((destination / "tests").exists())

    def test_launcher_and_path_file_use_isolated_official_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / APP_FOLDER_NAME
            runtime = candidate / "runtime"
            runtime.mkdir(parents=True)

            path_file = write_isolated_path_file(runtime)
            launcher = write_launcher(candidate)

            self.assertEqual(
                ".\nDLLs\nLib\nLib\\site-packages\n..\\app\n",
                path_file.read_text(encoding="utf-8"),
            )
            launcher_text = launcher.read_text(encoding="utf-8")
            self.assertIn("runtime\\pythonw.exe", launcher_text)
            self.assertIn("app\\app.py", launcher_text)
            self.assertIn("-E -s -X utf8", launcher_text)
            self.assertNotIn("nuitka", launcher_text.casefold())
            self.assertNotIn("pyinstaller", launcher_text.casefold())

    def test_release_zip_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = root / APP_FOLDER_NAME
            (candidate / "app").mkdir(parents=True)
            (candidate / "START_TOOL.cmd").write_text("start\n", encoding="utf-8")
            payload = candidate / "app" / "app.py"
            payload.write_text("print('test')\n", encoding="utf-8")
            first = root / "first.zip"
            second = root / "second.zip"

            create_deterministic_zip(candidate, first)
            payload.touch()
            create_deterministic_zip(candidate, second)

            first_hash = hashlib.sha256(first.read_bytes()).digest()
            second_hash = hashlib.sha256(second.read_bytes()).digest()
            self.assertEqual(first_hash, second_hash)

    def test_runtime_pruning_removes_build_only_executables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir) / "runtime"
            site_packages = runtime / "Lib" / "site-packages"
            (site_packages / "bin").mkdir(parents=True)
            metadata = site_packages / "cffi-2.1.1.dist-info"
            metadata.mkdir()
            (runtime / "Lib" / "venv").mkdir()
            (runtime / "Lib" / "ensurepip").mkdir()
            (runtime / "Doc").mkdir()
            (runtime / "include").mkdir()
            (runtime / "python.exe").write_bytes(b"official console runtime")
            (runtime / "pythonw.exe").write_bytes(b"official gui runtime")
            (site_packages / "bin" / "cffi-gen-src.exe").write_bytes(b"build tool")
            record = metadata / "RECORD"
            record.write_text(
                "../../bin/cffi-gen-src.exe,sha256=random,10\n"
                "cffi/__init__.py,sha256=stable,20\n",
                encoding="utf-8",
            )

            prune_runtime(runtime)

            self.assertTrue((runtime / "python.exe").is_file())
            self.assertTrue((runtime / "pythonw.exe").is_file())
            self.assertFalse((site_packages / "bin").exists())
            self.assertFalse((runtime / "Lib" / "venv").exists())
            self.assertFalse((runtime / "Lib" / "ensurepip").exists())
            self.assertFalse((runtime / "Doc").exists())
            self.assertFalse((runtime / "include").exists())
            self.assertEqual(
                "cffi/__init__.py,sha256=stable,20\n",
                record.read_text(encoding="utf-8"),
            )

    def test_native_compiler_can_target_candidate_directory(self) -> None:
        source = Path("native") / "pa_checksum.c"
        output = Path("candidate") / "app" / "crimson0020" / "native" / "pa_checksum.dll"

        command = compiler_command(source, output, "python.exe")

        self.assertEqual(("python.exe", "-m", "ziglang", "cc"), command[:4])
        self.assertIn("-s", command)
        self.assertNotIn("-Wl,--no-insert-timestamp", command)
        self.assertIn(str(output), command)
        self.assertEqual(str(source), command[-1])

    def test_generated_bytecode_is_removed_after_smoke_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "app" / "crimson0020"
            cache = package / "__pycache__"
            cache.mkdir(parents=True)
            source = package / "service.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            (cache / "service.cpython-314.pyc").write_bytes(b"compiled")
            loose = root / "runtime" / "module.pyc"
            loose.parent.mkdir()
            loose.write_bytes(b"compiled")

            remove_generated_bytecode(root)

            self.assertTrue(source.is_file())
            self.assertFalse(cache.exists())
            self.assertFalse(loose.exists())


if __name__ == "__main__":
    unittest.main()
