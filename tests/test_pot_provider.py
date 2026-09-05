"""Tests for app/analysis/pot_provider.py (the vendored bgutil-ytdlp-pot-
provider integration used only as a fallback retry when YouTube declines a
plain native download -- see docs/input_sources.md, "Proof-of-Origin token
support").

No network access and no real `deno install` is triggered by the default
suite: `ensure_provider_ready` is only exercised for its pure decision
logic (missing server dir / missing plugin / missing deno), never for a
real subprocess run, which is left to manual/integration verification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis import pot_provider


@pytest.fixture(autouse=True)
def _clear_memoized_readiness():
    pot_provider.ensure_provider_ready.cache_clear()
    yield
    pot_provider.ensure_provider_ready.cache_clear()


def test_server_home_is_absolute_and_not_home_relative():
    """Step 6: must not depend on `~`/home-directory assumptions -- derived
    from this file's own location instead."""
    server_home = pot_provider.SERVER_HOME
    assert server_home.is_absolute()
    assert "~" not in str(server_home)
    assert str(server_home).endswith(str(Path("third_party") / "bgutil-ytdlp-pot-provider" / "server"))


def test_provider_extractor_args_selects_mweb_client():
    args = pot_provider.provider_extractor_args()
    assert args["youtube"]["player_client"] == ["mweb"]


def test_provider_extractor_args_points_at_absolute_server_home():
    args = pot_provider.provider_extractor_args()
    server_home_arg = args["youtubepot-bgutilscript"]["server_home"][0]
    assert server_home_arg == str(pot_provider.SERVER_HOME)
    assert Path(server_home_arg).is_absolute()


def test_provider_extractor_args_never_includes_cookies_or_proxy_or_credentials():
    """Policy guard (docs/development_policy.md / docs/input_sources.md): this feature is
    public-content-only -- no cookies, no proxy, no account auth, ever."""
    import json

    serialized = json.dumps(pot_provider.provider_extractor_args()).lower()
    for forbidden in ("cookie", "proxy", "password", "oauth", "login"):
        assert forbidden not in serialized


def test_ensure_provider_ready_false_when_server_home_missing(monkeypatch):
    monkeypatch.setattr(pot_provider, "SERVER_HOME", Path("/nonexistent/does/not/exist/server"))
    ready, reason = pot_provider.ensure_provider_ready()
    assert ready is False
    assert reason


def test_ensure_provider_ready_false_when_plugin_not_importable(monkeypatch):
    monkeypatch.setattr(pot_provider, "_plugin_importable", lambda: False)
    ready, reason = pot_provider.ensure_provider_ready()
    assert ready is False
    assert reason


def test_ensure_provider_ready_false_when_deno_unavailable(monkeypatch):
    monkeypatch.setattr(pot_provider, "_plugin_importable", lambda: True)
    monkeypatch.setattr(pot_provider, "_deno_bin", lambda: None)
    ready, reason = pot_provider.ensure_provider_ready()
    assert ready is False
    assert reason


def test_ensure_provider_ready_skips_install_when_deps_already_present(monkeypatch):
    """If node_modules already exists (e.g. a previous process already
    installed it), no subprocess should be spawned at all."""
    monkeypatch.setattr(pot_provider, "_plugin_importable", lambda: True)
    monkeypatch.setattr(pot_provider, "_deno_bin", lambda: "/usr/bin/deno")
    monkeypatch.setattr(pot_provider, "_native_deps_installed", lambda: True)

    called = {"n": 0}

    def _should_not_run(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("subprocess.run should not be called when deps are already installed")

    monkeypatch.setattr(pot_provider.subprocess, "run", _should_not_run)
    ready, reason = pot_provider.ensure_provider_ready()
    assert ready is True
    assert reason == ""
    assert called["n"] == 0


def test_ensure_provider_ready_is_memoized_across_calls(monkeypatch):
    """The (potentially slow) readiness check must run at most once per
    process -- step 17, resource safety."""
    monkeypatch.setattr(pot_provider, "_plugin_importable", lambda: True)
    monkeypatch.setattr(pot_provider, "_deno_bin", lambda: "/usr/bin/deno")
    monkeypatch.setattr(pot_provider, "_native_deps_installed", lambda: True)

    pot_provider.ensure_provider_ready()
    pot_provider.ensure_provider_ready()
    pot_provider.ensure_provider_ready()
    assert pot_provider.ensure_provider_ready.cache_info().hits >= 2


def test_ensure_provider_ready_reports_failure_when_install_exits_nonzero(monkeypatch):
    monkeypatch.setattr(pot_provider, "_plugin_importable", lambda: True)
    monkeypatch.setattr(pot_provider, "_deno_bin", lambda: "/usr/bin/deno")
    monkeypatch.setattr(pot_provider, "_native_deps_installed", lambda: False)
    monkeypatch.setattr(pot_provider, "SERVER_HOME", Path(__file__).parent)  # any existing dir

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "some npm install error"

    monkeypatch.setattr(pot_provider.subprocess, "run", lambda *a, **k: _Proc())
    ready, reason = pot_provider.ensure_provider_ready()
    assert ready is False
    assert reason
