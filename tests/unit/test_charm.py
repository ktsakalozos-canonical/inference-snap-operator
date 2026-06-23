# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.
"""Unit tests for the inference-snap charm."""

from unittest.mock import MagicMock, patch

import pytest
import ops
from ops import testing

import charm


class FakeSnapError(Exception):
    """Stand-in for snap.SnapError."""


class FakeSnapNotFoundError(Exception):
    """Stand-in for snap.SnapNotFoundError."""


@pytest.fixture
def ctx():
    return testing.Context(charm.InferenceSnapCharm)


@pytest.fixture
def snap_cache():
    """Patch the snap module used by the charm.

    Returns a dict mapping snap name -> MagicMock Snap, and the SnapCache mock
    behaves like a dict over that mapping.
    """
    snaps: dict[str, MagicMock] = {}

    def _make(name):
        if name not in snaps:
            m = MagicMock(name=f"snap:{name}")
            m.present = False
            snaps[name] = m
        return snaps[name]

    cache = MagicMock()
    cache.__getitem__.side_effect = _make

    with patch.object(charm, "snap") as snap_mod:
        snap_mod.SnapCache.return_value = cache
        snap_mod.SnapError = FakeSnapError
        snap_mod.SnapNotFoundError = FakeSnapNotFoundError
        snap_mod.SnapState.Present = "present"
        snap_mod.SnapState.Absent = "absent"
        yield snaps


@pytest.fixture(autouse=True)
def mock_subprocess():
    """Stub out the snap CLI calls (`<snap> set ...`, `<snap> status`).

    Tests that need a specific ``status`` output set ``return_value.stdout``
    or use their own ``@patch`` decorator, which takes precedence.
    """
    with patch("charm.subprocess.run") as run:
        run.return_value = MagicMock(stdout="", returncode=0)
        yield run


def test_invalid_snap_blocks(ctx, snap_cache):
    state_in = testing.State(config={"inference-snap": "not-a-real-snap"})
    state_out = ctx.run(ctx.on.config_changed(), state_in)
    assert isinstance(state_out.unit_status, ops.BlockedStatus)
    assert "Unknown inference snap" in state_out.unit_status.message


def test_default_snap_is_gemma3(ctx, snap_cache):
    state_out = ctx.run(ctx.on.install(), testing.State())
    assert "gemma3" in snap_cache
    snap_cache["gemma3"].ensure.assert_called_once()
    _, kwargs = snap_cache["gemma3"].ensure.call_args
    assert kwargs["channel"] == "latest/stable"
    assert isinstance(state_out.unit_status, ops.ActiveStatus)
    assert "gemma3" in state_out.unit_status.message


def test_custom_channel_passed_through(ctx, snap_cache):
    state_in = testing.State(
        config={"inference-snap": "qwen3", "snap-channel": "latest/beta"}
    )
    ctx.run(ctx.on.config_changed(), state_in)
    _, kwargs = snap_cache["qwen3"].ensure.call_args
    assert kwargs["channel"] == "latest/beta"


def test_http_config_applied(ctx, snap_cache, mock_subprocess):
    state_in = testing.State(config={"api-port": 9090, "api-bind-all": True})
    ctx.run(ctx.on.config_changed(), state_in)
    # Configuration goes through the snap's own CLI, not snapd.
    args = mock_subprocess.call_args.args[0]
    assert args[0] == "gemma3"
    assert args[1] == "set"
    assert "http.host=0.0.0.0" in args
    assert "http.port=9090" in args
    # The snap service is (re)enabled afterwards.
    snap_cache["gemma3"].start.assert_called_with(enable=True)


def test_bind_local_only(ctx, snap_cache, mock_subprocess):
    state_in = testing.State(config={"api-bind-all": False})
    ctx.run(ctx.on.config_changed(), state_in)
    args = mock_subprocess.call_args.args[0]
    assert "http.host=127.0.0.1" in args


def test_port_opened(ctx, snap_cache):
    state_in = testing.State(config={"api-port": 9091})
    state_out = ctx.run(ctx.on.config_changed(), state_in)
    assert testing.TCPPort(9091) in state_out.opened_ports


def test_port_changes_close_old(ctx, snap_cache):
    state_in = testing.State(
        config={"api-port": 9091},
        opened_ports={testing.TCPPort(8080)},
    )
    state_out = ctx.run(ctx.on.config_changed(), state_in)
    ports = {p.port for p in state_out.opened_ports}
    assert ports == {9091}


def test_switching_snaps_removes_previous(ctx, snap_cache):
    # Pretend gemma3 is already installed.
    pre = snap_cache  # snaps dict; trigger creation of gemma3 as present
    cache_snap = MagicMock()
    cache_snap.present = True
    pre["gemma3"] = cache_snap

    state_in = testing.State(config={"inference-snap": "qwen3"})
    ctx.run(ctx.on.config_changed(), state_in)

    # Old snap removed, new snap installed.
    pre["gemma3"].ensure.assert_called_with("absent")
    qwen_calls = pre["qwen3"].ensure.call_args
    assert qwen_calls.kwargs.get("channel") == "latest/stable"


def test_snap_error_blocks(ctx, snap_cache):
    snap_cache_dict = snap_cache

    def boom(*a, **k):
        raise FakeSnapError("install failed")

    # Pre-create gemma3 mock so ensure raises.
    m = MagicMock()
    m.present = False
    m.ensure.side_effect = boom
    snap_cache_dict["gemma3"] = m

    state_out = ctx.run(ctx.on.install(), testing.State())
    assert isinstance(state_out.unit_status, ops.BlockedStatus)
    assert "Snap operation failed" in state_out.unit_status.message


@patch("charm.subprocess.run")
def test_endpoint_published_on_relation(mock_run, ctx, snap_cache):
    mock_run.return_value = MagicMock(
        stdout="OpenAI API: http://localhost:8080/v1/engine"
    )
    # Make gemma3 report present so the relation handler publishes.
    gemma = MagicMock()
    gemma.present = True
    snap_cache["gemma3"] = gemma

    relation = testing.Relation(endpoint="inference-api", interface="inference_openai")
    state_in = testing.State(leader=True, relations={relation})
    state_out = ctx.run(ctx.on.config_changed(), state_in)

    rel_out = state_out.get_relation(relation.id)
    app_data = rel_out.local_app_data
    assert app_data["snap"] == "gemma3"
    assert app_data["model"] == "gemma3"
    assert app_data["port"] == "8080"
    assert app_data["url"].endswith("/v1/engine")
    # Leader also publishes its own unit databag.
    assert rel_out.local_unit_data["url"].endswith("/v1/engine")


@patch("charm.subprocess.run")
def test_endpoint_prefers_openai_label_over_first_url(mock_run, ctx, snap_cache):
    # Engines like intel-cpu advertise multiple endpoints; the OpenAI base path
    # is NOT the first URL listed. We must select the ``openai`` endpoint.
    mock_run.return_value = MagicMock(
        stdout=(
            "kserve: http://localhost:8080/v2\n"
            "openai: http://localhost:8080/v3\n"
            "tensorflow-serving: http://localhost:8080/v1\n"
            "webui: http://localhost:8329/\n"
        )
    )
    gemma = MagicMock()
    gemma.present = True
    snap_cache["gemma3"] = gemma

    relation = testing.Relation(endpoint="inference-api", interface="inference_openai")
    state_in = testing.State(leader=True, relations={relation})
    state_out = ctx.run(ctx.on.config_changed(), state_in)

    rel_out = state_out.get_relation(relation.id)
    assert rel_out.local_app_data["url"].endswith("/v3")
    assert "/v2" not in rel_out.local_app_data["url"]


@patch("charm.subprocess.run")
def test_endpoint_published_per_unit_when_not_leader(mock_run, ctx, snap_cache):
    mock_run.return_value = MagicMock(
        stdout="OpenAI API: http://localhost:8080/v1/engine"
    )
    gemma = MagicMock()
    gemma.present = True
    snap_cache["gemma3"] = gemma

    relation = testing.Relation(endpoint="inference-api", interface="inference_openai")
    state_in = testing.State(leader=False, relations={relation})
    state_out = ctx.run(ctx.on.config_changed(), state_in)

    rel_out = state_out.get_relation(relation.id)
    # Non-leaders must not write app data, but must publish their own unit data
    # so the router can load-balance across all units.
    assert dict(rel_out.local_app_data) == {}
    unit_data = rel_out.local_unit_data
    assert unit_data["snap"] == "gemma3"
    assert unit_data["model"] == "gemma3"
    assert unit_data["port"] == "8080"
    assert unit_data["url"].endswith("/v1/engine")
