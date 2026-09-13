import json
import os
import pytest
from crimson0020.translation_runtime.contracts import (
    FailureKind,
    TranslationBatchRequest,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.router.adapters.cloudflare import CloudflareAdapter
from crimson0020.translation_runtime.router.browser_worker import (
    EdgeDiscoveryError,
    IsolatedBrowserWorker,
    find_system_edge_executable,
)

def test_system_edge_discovery_missing(monkeypatch):
    monkeypatch.setenv("EDGE_PATH", "/nonexistent/msedge.exe")
    edge = find_system_edge_executable(custom_path="/nonexistent/msedge.exe")
    assert edge is None

def test_system_edge_discovery_env_and_path(tmp_path, monkeypatch):
    fake_edge = tmp_path / "msedge.exe"
    fake_edge.touch()

    # Discovery via custom path
    found = find_system_edge_executable(custom_path=fake_edge)
    assert found == fake_edge.resolve()

    # Discovery via EDGE_PATH env
    monkeypatch.setenv("EDGE_PATH", str(fake_edge))
    found_env = find_system_edge_executable()
    assert found_env == fake_edge.resolve()

    # Discovery via PATH env
    monkeypatch.delenv("EDGE_PATH", raising=False)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}C:\\Windows")
    found_path = find_system_edge_executable()
    assert found_path == fake_edge.resolve()

def test_isolated_browser_worker_fake_cdp(tmp_path):
    def fake_cdp_runner(command, port, profile_dir):
        assert port > 1024
        assert profile_dir.exists()
        return {
            "status": "success",
            "result": json.dumps({
                "translations": [
                    {"id": "cf1", "text": "Dịch Cloudflare"}
                ]
            }, ensure_ascii=False)
        }

    worker = IsolatedBrowserWorker(
        runner=fake_cdp_runner,
        profile_root=tmp_path / "profiles",
    )
    res = worker.execute_inference(
        prompt="Hello",
        allowed_origin="ai.cloudflare.com",
    )
    assert res["status"] == "success"
    assert "Dịch Cloudflare" in res["result"]

def test_isolated_browser_worker_security_checks(tmp_path):
    worker = IsolatedBrowserWorker(profile_root=tmp_path / "profiles")
    with pytest.raises(ValueError):
        worker.execute_inference(prompt="Hello", allowed_origin="malicious.attacker.com")

def test_isolated_browser_worker_missing_executable_error(tmp_path):
    worker = IsolatedBrowserWorker(edge_executable=tmp_path / "absent.exe", profile_root=tmp_path / "p")
    with pytest.raises(EdgeDiscoveryError):
        worker.execute_inference(prompt="Hello")

def test_isolated_browser_worker_default_execution(tmp_path):
    fake_edge = tmp_path / "msedge.exe"
    fake_edge.touch()
    worker = IsolatedBrowserWorker(edge_executable=fake_edge, profile_root=tmp_path / "p")
    res = worker.execute_inference(prompt="Hello")
    assert res["status"] == "unavailable"

def test_cloudflare_adapter_graceful_unavailable():
    adapter = CloudflareAdapter(edge_path="/nonexistent/msedge.exe")
    batch = TranslationBatchRequest(
        job_id="j1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="cf1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter.translate_batch(batch)
    assert len(outcome.failures) == 1
    assert outcome.failures[0].kind in (FailureKind.UNSUPPORTED_MODEL, FailureKind.PROVIDER_INTERNAL)
    assert outcome.failures[0].retryable is False

def test_cloudflare_adapter_success_with_worker(tmp_path):
    def fake_cdp_runner(command, port, profile_dir):
        return {
            "status": "success",
            "result": json.dumps({
                "translations": [
                    {"id": "cf1", "text": "Dịch Cloudflare AI"}
                ]
            }, ensure_ascii=False)
        }
    worker = IsolatedBrowserWorker(runner=fake_cdp_runner, profile_root=tmp_path / "profiles")
    adapter = CloudflareAdapter(worker=worker)
    batch = TranslationBatchRequest(
        job_id="j1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="cf1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter.translate_batch(batch)
    assert len(outcome.successes) == 1
    assert outcome.successes[0].translated_text == "Dịch Cloudflare AI"

def test_cloudflare_adapter_worker_error_and_empty_batch(tmp_path):
    adapter = CloudflareAdapter(edge_path="/nonexistent/msedge.exe")
    empty = TranslationBatchRequest(job_id="j", source_language="en", target_language="vi", items=())
    assert len(adapter.translate_batch(empty).successes) == 0

    def fake_fail_runner(command, port, profile_dir):
        return {"status": "error", "error": "CDP connection lost"}
    worker_fail = IsolatedBrowserWorker(runner=fake_fail_runner, profile_root=tmp_path / "profiles")
    adapter_fail = CloudflareAdapter(worker=worker_fail)
    batch = TranslationBatchRequest(
        job_id="j",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="cf1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome_fail = adapter_fail.translate_batch(batch)
    assert outcome_fail.failures[0].kind == FailureKind.PROVIDER_INTERNAL
