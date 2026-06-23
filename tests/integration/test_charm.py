# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.
"""Integration tests for the inference-snap charm."""

import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

APP_NAME = "inference-snap"


@pytest.mark.abort_on_fail
async def test_build_and_deploy(ops_test):
    """Build the charm and deploy it, expecting it to reach active.

    Note: a real deployment installs a multi-gigabyte model snap, so this test
    targets the smallest Gemma snap from the beta channel and allows a long
    timeout.
    """
    charm = await ops_test.build_charm(".")
    assert Path(str(charm)).exists()

    await ops_test.model.deploy(
        charm,
        application_name=APP_NAME,
        config={"inference-snap": "gemma3", "snap-channel": "latest/beta"},
        base="ubuntu@22.04",
    )
    await ops_test.model.wait_for_idle(apps=[APP_NAME], status="active", timeout=3600)


@pytest.mark.abort_on_fail
async def test_invalid_snap_blocks(ops_test):
    """An unknown snap name should block the unit."""
    await ops_test.model.applications[APP_NAME].set_config(
        {"inference-snap": "definitely-not-a-snap"}
    )
    await ops_test.model.wait_for_idle(apps=[APP_NAME], status="blocked", timeout=300)
    # Restore so subsequent tests/teardown are clean.
    await ops_test.model.applications[APP_NAME].set_config({"inference-snap": "gemma3"})
