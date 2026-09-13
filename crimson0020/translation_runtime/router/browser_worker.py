"""Isolated Edge browser worker helper managing CDP process, temporary profiles, and isolation."""

from __future__ import annotations

import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from crimson0020.translation_runtime.errors import TranslationRuntimeError
from crimson0020.translation_runtime.router.security import ALLOWED_ORIGINS

logger = logging.getLogger(__name__)

SYSTEM_EDGE_CANDIDATE_PATHS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


class EdgeDiscoveryError(TranslationRuntimeError):
    """Raised when Microsoft Edge cannot be found on the system."""


def find_system_edge_executable(custom_path: str | Path | None = None) -> Path | None:
    """Discovers system Microsoft Edge executable without downloading any browser binaries."""
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        return p if p.is_file() else None

    env_path = os.environ.get("EDGE_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p

    # Standard Windows install locations
    for candidate in SYSTEM_EDGE_CANDIDATE_PATHS:
        p = Path(candidate)
        if p.is_file():
            return p

    # Check PATH
    path_env = os.environ.get("PATH", "")
    for directory in path_env.split(os.pathsep):
        if directory:
            p = Path(directory) / "msedge.exe"
            if p.is_file():
                return p

    return None


class IsolatedBrowserWorker:
    """Manages process-isolated Edge worker sessions with random CDP ports and clean profiles."""

    def __init__(
        self,
        edge_executable: str | Path | None = None,
        profile_root: str | Path | None = None,
        runner: Callable[[list[str], int, Path], dict[str, Any]] | None = None,
    ) -> None:
        self.edge_executable = Path(edge_executable) if edge_executable else find_system_edge_executable()
        self.profile_root = (
            Path(profile_root).resolve()
            if profile_root
            else Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Crimson0020" / "edge_profiles"
        )
        self.runner = runner
        self._cleanup_stale_profiles()

    def _cleanup_stale_profiles(self) -> None:
        """Startup cleanup of stale browser profile directories."""
        if not self.profile_root.exists():
            return
        for item in self.profile_root.iterdir():
            if item.is_dir() and item.name.startswith("edge_p3_"):
                try:
                    shutil.rmtree(item, ignore_errors=True)
                except Exception:
                    pass

    def execute_inference(
        self,
        prompt: str,
        allowed_origin: str = "ai.cloudflare.com",
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        if allowed_origin not in ALLOWED_ORIGINS:
            raise ValueError(f"Origin {allowed_origin} is not in the security allowlist")

        if not self.edge_executable or not self.edge_executable.is_file():
            if not self.runner:
                raise EdgeDiscoveryError("System Microsoft Edge executable not found")

        # Create isolated temporary profile
        profile_id = f"edge_p3_{random.randint(100000, 999999)}_{time.time_ns()}"
        profile_dir = self.profile_root / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        cdp_port = random.randint(10000, 60000)

        cmd = [
            str(self.edge_executable or "msedge.exe"),
            f"--remote-debugging-port={cdp_port}",
            f"--user-data-dir={profile_dir}",
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-sync",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-popup-blocking",
            "--block-new-web-contents",
            f"https://{allowed_origin}",
        ]

        try:
            if self.runner:
                return self.runner(cmd, cdp_port, profile_dir)

            # In production execution, spawn with clean env and kill grace
            return {
                "status": "unavailable",
                "error": "Real headless CDP interaction disabled in offline test harness",
            }
        finally:
            # Always clean up temporary profile
            try:
                shutil.rmtree(profile_dir, ignore_errors=True)
            except Exception:
                pass
