import tempfile
import unittest
from pathlib import Path

from scripts.verify_public_source import collect_violations


class PublicSourceTests(unittest.TestCase):
    def test_public_readmes_describe_current_features_and_build_only(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        readme_vi = (project_root / "README.md").read_text(encoding="utf-8")
        readme_en = (project_root / "README_ENG.md").read_text(encoding="utf-8")

        self.assertIn("[English](README_ENG.md)", readme_vi)
        self.assertIn("[Tiếng Việt](README.md)", readme_en)
        for content in (readme_vi, readme_en):
            self.assertIn(
                "python -m pip install --no-deps -r requirements-build.txt",
                content,
            )
            self.assertIn("python scripts/build_onedir.py --output-root", content)
            for historical_term in (
                "nuitka",
                "pyinstaller",
                "upx",
                "freezer",
                "pipeline nghiên cứu cũ",
                "legacy research pipeline",
            ):
                self.assertNotIn(historical_term, content.casefold())

    def test_v2_release_build_entrypoints_are_configured(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        build_cmd = (project_root / "BUILD_EXE.cmd").read_text(encoding="utf-8")
        release_workflow = (
            project_root / ".github" / "workflows" / "windows-release.yml"
        ).read_text(encoding="utf-8")
        ci_workflow = (
            project_root / ".github" / "workflows" / "windows-ci.yml"
        ).read_text(encoding="utf-8")
        package_init = (project_root / "crimson0020" / "__init__.py").read_text(
            encoding="utf-8"
        )
        readme_vi = (project_root / "README.md").read_text(encoding="utf-8")
        readme_en = (project_root / "README_ENG.md").read_text(encoding="utf-8")

        self.assertIn("requirements-build.txt", build_cmd)
        self.assertIn(r"scripts\build_onedir.py", build_cmd)
        self.assertIn("release_onedir_v2.0.1", build_cmd)
        self.assertIn('tags:\n      - "v2.0.1"', release_workflow)
        self.assertIn("contents: write", release_workflow)
        self.assertIn("python scripts/build_onedir.py", release_workflow)
        self.assertIn("CrimsonDesertLocalizationBuilder-windows-x64.zip", release_workflow)
        self.assertIn("SHA256SUMS.txt", release_workflow)
        self.assertIn("gh release create", release_workflow)
        self.assertIn("branches:\n      - main", ci_workflow)
        self.assertIn('__version__ = "2.0.1"', package_init)
        self.assertIn(r".\BUILD_EXE.cmd", readme_vi)
        self.assertIn(r".\BUILD_EXE.cmd", readme_en)

    def test_build_requirements_include_complete_pytest_closure(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        requirements = (project_root / "requirements-build.txt").read_text(
            encoding="utf-8"
        )

        for requirement in (
            "pytest==9.1.1",
            "iniconfig==2.3.0",
            "packaging==26.3",
            "pluggy==1.6.0",
            "pygments==2.21.0",
        ):
            self.assertIn(requirement, requirements.casefold())

    def test_clean_source_tree_has_no_violations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text("print('safe')\n", encoding="utf-8")
            (root / "README.md").write_text("Public source\n", encoding="utf-8")

            self.assertEqual((), collect_violations(root))

    def test_only_the_declared_public_icon_assets_may_be_binary_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = root / "crimson0020" / "assets"
            assets.mkdir(parents=True)
            (assets / "crimson-desert-localization.ico").write_bytes(b"\x00\x01")
            (assets / "crimson-desert-localization.png").write_bytes(b"\x89PNG")

            self.assertEqual((), collect_violations(root))

            (assets / "unapproved.png").write_bytes(b"\x89PNG")
            violations = collect_violations(root)

            self.assertTrue(any("unapproved.png" in item for item in violations))

    def test_game_archives_and_generated_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output_repack" / "0020"
            output.mkdir(parents=True)
            (output / "0.paz").write_bytes(b"game archive")

            violations = collect_violations(root)

            self.assertTrue(any("output_repack" in item for item in violations))
            self.assertTrue(any(".paz" in item for item in violations))

    def test_personal_paths_and_secret_shaped_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            personal_path = "C:" + "\\Users\\example\\private"
            fake_token = "github_" + "pat_" + ("A" * 32)
            (root / "notes.txt").write_text(
                f"path={personal_path}\ntoken={fake_token}\n",
                encoding="utf-8",
            )

            violations = collect_violations(root)

            self.assertTrue(any("personal path" in item for item in violations))
            self.assertTrue(any("credential-like" in item for item in violations))

    def test_local_translation_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in (
                "custom_openai_config.json",
                "translation-knowledge-v1.sqlite3-wal",
                "work.autotranslate-checkpoint.jsonl",
                "work.autotranslate-failures.jsonl",
            ):
                (root / name).write_text("local runtime data\n", encoding="utf-8")

            violations = collect_violations(root)

            for name in (
                "custom_openai_config.json",
                "translation-knowledge-v1.sqlite3-wal",
                "work.autotranslate-checkpoint.jsonl",
                "work.autotranslate-failures.jsonl",
            ):
                self.assertTrue(any(name in item for item in violations))

    def test_github_workflow_verifies_clean_source_and_tests(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        workflow_path = project_root / ".github" / "workflows" / "windows-ci.yml"

        self.assertTrue(workflow_path.is_file())
        workflow = workflow_path.read_text(encoding="utf-8")
        for required_text in (
            "pull_request:",
            "contents: read",
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
            "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
            'python-version: "3.13"',
            "python scripts/verify_public_source.py .",
            "python -m pip install --no-deps -r requirements-build.txt",
            "python -m unittest discover -s tests -v",
            "python build_native.py",
            "python -m compileall -q app.py sitecustomize.py crimson0020 scripts tests",
        ):
            self.assertIn(required_text, workflow)
        self.assertLess(
            workflow.index("python build_native.py"),
            workflow.index("python -m unittest discover -s tests -v"),
        )
        self.assertNotIn("secrets.", workflow)
        self.assertNotIn("nuitka", workflow.casefold())
        self.assertNotIn("upload-artifact", workflow.casefold())

    def test_build_entrypoint_uses_current_onedir_builder(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        requirements = (project_root / "requirements-build.txt").read_text(
            encoding="utf-8"
        )
        workflow = (
            project_root / ".github" / "workflows" / "windows-ci.yml"
        ).read_text(encoding="utf-8")
        gui_paths = (project_root / "crimson0020" / "gui_paths.py").read_text(
            encoding="utf-8"
        )
        build_cmd = (project_root / "BUILD_EXE.cmd").read_text(encoding="utf-8")

        self.assertIn("-r requirements.txt", requirements)
        self.assertIn("ziglang==0.16.0", requirements)
        self.assertNotIn("nuitka", requirements.casefold())
        self.assertNotIn("nuitka", workflow.casefold())
        self.assertNotIn("pyinstaller", requirements.casefold())
        self.assertNotIn("cx-freeze", requirements.casefold())
        self.assertNotIn("__compiled__", gui_paths)
        self.assertIn(r"scripts\build_onedir.py", build_cmd)
        self.assertNotIn("nuitka", build_cmd.casefold())
        self.assertNotIn("pyinstaller", build_cmd.casefold())
        self.assertFalse((project_root / "nuitka-package.config.yml").exists())

    def test_portable_build_uses_official_python_without_a_freezer(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        build_script = (project_root / "scripts" / "build_portable.py").read_text(
            encoding="utf-8"
        )
        lockfile = (project_root / "requirements-portable.txt").read_text(
            encoding="utf-8"
        )
        workflow = (
            project_root / ".github" / "workflows" / "windows-ci.yml"
        ).read_text(encoding="utf-8")

        for forbidden in ("nuitka", "pyinstaller", "cx_freeze", "briefcase", "upx"):
            self.assertNotIn(forbidden, build_script.casefold())
        self.assertIn("python-3.14.7-amd64.zip", build_script)
        self.assertIn("verify_runtime_archive", build_script)
        self.assertIn("requirements-portable.txt", build_script)
        self.assertIn("--require-hashes", build_script)
        for package in (
            "cryptography",
            "lz4",
            "cffi",
            "pycparser",
            "pywebview",
            "pythonnet",
            "clr_loader",
            "bottle",
            "typing_extensions",
            "websocket-client",
            "pygtrans",
            "translators",
            "lxml",
        ):
            self.assertRegex(lockfile, rf"(?m)^{package}==[^\s]+ \\")
        self.assertEqual(47, lockfile.count("--hash=sha256:"))
        self.assertIn("--no-deps", build_script)
        self.assertNotIn("--only-binary", build_script)
        self.assertTrue((project_root / "proxy_tools" / "__init__.py").is_file())
        self.assertTrue((project_root / "cloudscraper" / "__init__.py").is_file())
        self.assertNotRegex(lockfile, r"(?mi)^js2py==")
        self.assertNotRegex(lockfile, r"(?mi)^ai[_-]cloudscraper==")
        self.assertTrue((project_root / "THIRD_PARTY_NOTICES.md").is_file())
        self.assertIn("python scripts/build_onedir.py", workflow)
        self.assertNotIn("upload-artifact", workflow.casefold())

    def test_one_folder_build_uses_transparent_microsoft_launcher(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        build_script = (project_root / "scripts" / "build_onedir.py").read_text(
            encoding="utf-8"
        )
        launcher_builder = (project_root / "build_launcher.py").read_text(
            encoding="utf-8"
        )
        launcher_source = (project_root / "launcher" / "ToolLauncher.cs").read_text(
            encoding="utf-8"
        )
        workflow = (
            project_root / ".github" / "workflows" / "windows-ci.yml"
        ).read_text(encoding="utf-8")

        combined = (build_script + launcher_builder + launcher_source).casefold()
        for forbidden in ("nuitka", "pyinstaller", "cx_freeze", "briefcase", "upx"):
            self.assertNotIn(forbidden, combined)
        self.assertIn("Crimson.Desert.Localization.Builder.exe", build_script)
        self.assertIn("compile_launcher", build_script)
        self.assertIn("/deterministic+", launcher_builder)
        self.assertIn("Microsoft.NET", launcher_builder)
        self.assertIn("ProcessStartInfo", launcher_source)
        self.assertIn('"_internal", "runtime", "pythonw.exe"', launcher_source)
        self.assertTrue((project_root / "build_launcher.py").is_file())
        self.assertFalse((project_root / "native" / "tool_launcher.c").exists())
        self.assertIn("python scripts/build_onedir.py", workflow)
        self.assertNotIn("upload-artifact", workflow.casefold())

    def test_self_extracting_cached_packaging_is_removed(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        forbidden_paths = (
            project_root / "scripts" / "build_cached.py",
            project_root / "launcher" / "CachedPayload.cs",
            project_root / "tests" / "test_cached_payload.py",
        )
        for path in forbidden_paths:
            self.assertFalse(path.exists(), f"Obsolete cached packaging remains: {path}")

        build_script = (project_root / "scripts" / "build_onedir.py").read_text(
            encoding="utf-8"
        )
        launcher_source = (project_root / "launcher" / "ToolLauncher.cs").read_text(
            encoding="utf-8"
        )
        launcher_builder = (project_root / "build_launcher.py").read_text(
            encoding="utf-8"
        )
        combined = build_script + launcher_source + launcher_builder
        for forbidden_text in (
            "CACHED_PAYLOAD",
            "CachedPayload",
            "Crimson.Payload",
            "payload.zip",
            "System.IO.Compression.dll",
        ):
            self.assertNotIn(forbidden_text, combined)

    def test_gitignore_blocks_private_and_generated_files(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")

        for required_pattern in (
            ".venv/",
            "__pycache__/",
            ".coverage",
            "build/",
            "release/",
            "output_*/",
            "*.paz",
            "*.pamt",
            "*.papgt",
            "*.ttf",
            "*.exe",
            "*.dll",
            "config.json",
            "custom_openai_config.json",
            "translation-knowledge-v1*.sqlite3*",
            "*.autotranslate-checkpoint.jsonl",
            "*.autotranslate-failures.jsonl",
        ):
            self.assertIn(required_pattern, gitignore)


if __name__ == "__main__":
    unittest.main()
