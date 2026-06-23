#!/usr/bin/env python3
# Copyright 2026 Canonical Ltd.
# See LICENSE file for licensing details.

"""Charm for deploying and operating a Canonical inference snap.

The charm installs one of Canonical's optimized inference snaps
(https://documentation.ubuntu.com/inference-snaps/), configures its
OpenAI-compatible HTTP API, opens the corresponding port, and publishes the
resulting endpoint over the ``inference-api`` relation.
"""

import logging
import re
import socket
import subprocess
from typing import Optional

import ops
from charms.operator_libs_linux.v2 import snap

logger = logging.getLogger(__name__)

# Inference snaps published by Canonical.
# https://documentation.ubuntu.com/inference-snaps/reference/snaps/
KNOWN_INFERENCE_SNAPS = frozenset(
    {
        "gemma3",
        "gemma4",
        "deepseek-r1",
        "nemotron-3-nano",
        "nemotron-3-nano-omni",
        "qwen-vl",
        "qwen3",
        "qwen3-coder",
        "qwen3-6",
    }
)

DEFAULT_SNAP = "gemma3"
API_RELATION = "inference-api"


class InferenceSnapCharm(ops.CharmBase):
    """Operate a Canonical inference snap on a machine."""

    def __init__(self, framework: ops.Framework):
        super().__init__(framework)
        framework.observe(self.on.install, self._reconcile)
        framework.observe(self.on.config_changed, self._reconcile)
        framework.observe(self.on.upgrade_charm, self._reconcile)
        framework.observe(self.on.start, self._reconcile)
        framework.observe(self.on.remove, self._on_remove)
        framework.observe(
            self.on[API_RELATION].relation_joined, self._on_api_relation_changed
        )
        framework.observe(
            self.on[API_RELATION].relation_changed, self._on_api_relation_changed
        )

    # -- Event handlers ----------------------------------------------------

    def _reconcile(self, event: ops.EventBase) -> None:
        """Idempotently bring the unit in line with its configuration."""
        wanted = self._wanted_snap()
        if wanted is None:
            configured = self.config["inference-snap"]
            self.unit.status = ops.BlockedStatus(
                f"Unknown inference snap '{configured}'. "
                f"Choose one of: {', '.join(sorted(KNOWN_INFERENCE_SNAPS))}"
            )
            return

        try:
            self._install_snap(wanted)
            self._configure_snap(wanted)
        except snap.SnapError as e:
            logger.exception("Snap operation failed")
            self.unit.status = ops.BlockedStatus(f"Snap operation failed: {e}")
            return
        except (subprocess.SubprocessError, OSError) as e:
            logger.exception("Snap configuration failed")
            self.unit.status = ops.BlockedStatus(f"Snap configuration failed: {e}")
            return

        self._reconcile_ports()
        self._publish_endpoint(wanted)
        self.unit.status = ops.ActiveStatus(f"serving {wanted} on :{self._api_port()}")

    def _on_remove(self, event: ops.RemoveEvent) -> None:
        """Stop and remove the managed snap when the unit goes away."""
        installed = self._installed_inference_snap()
        if installed is None:
            return
        try:
            cache = snap.SnapCache()
            cache[installed].ensure(snap.SnapState.Absent)
        except snap.SnapError:
            logger.warning("Failed to remove snap %s during teardown", installed)

    def _on_api_relation_changed(self, event: ops.RelationEvent) -> None:
        """Publish the endpoint when a client relates to us."""
        wanted = self._wanted_snap()
        if wanted is not None and self._snap(wanted).present:
            self._publish_endpoint(wanted)

    # -- Configuration helpers --------------------------------------------

    def _wanted_snap(self) -> Optional[str]:
        """Return the validated configured snap name, or None if invalid."""
        name = str(self.config["inference-snap"]).strip()
        return name if name in KNOWN_INFERENCE_SNAPS else None

    def _api_port(self) -> int:
        return int(self.config["api-port"])

    def _api_host(self) -> str:
        return "0.0.0.0" if bool(self.config["api-bind-all"]) else "127.0.0.1"

    def _snap(self, name: str) -> snap.Snap:
        return snap.SnapCache()[name]

    def _installed_inference_snap(self) -> Optional[str]:
        """Return the name of whichever known inference snap is installed."""
        cache = snap.SnapCache()
        for name in KNOWN_INFERENCE_SNAPS:
            try:
                if cache[name].present:
                    return name
            except snap.SnapNotFoundError:
                continue
        return None

    # -- Snap lifecycle ----------------------------------------------------

    def _install_snap(self, wanted: str) -> None:
        """Install the wanted snap, removing any other inference snap first."""
        self.unit.status = ops.MaintenanceStatus(f"Installing {wanted}")
        channel = str(self.config["snap-channel"]).strip() or "latest/stable"

        # If the user switched snaps, remove the previously installed one.
        previous = self._installed_inference_snap()
        if previous is not None and previous != wanted:
            logger.info("Removing previous inference snap %s", previous)
            self._snap(previous).ensure(snap.SnapState.Absent)

        self._snap(wanted).ensure(snap.SnapState.Present, channel=channel)

    def _configure_snap(self, wanted: str) -> None:
        """Apply HTTP host/port config via the snap's own CLI and start it.

        Inference snaps ship no snapd ``configure`` hook, so ``snap set`` is
        rejected. They are instead configured through their bundled command,
        e.g. ``gemma3 set http.host=0.0.0.0 http.port=8080``. That command also
        restarts the snap's services so the new binding takes effect.
        """
        self.unit.status = ops.MaintenanceStatus(f"Configuring {wanted}")
        subprocess.run(
            [
                wanted,
                "set",
                f"http.host={self._api_host()}",
                f"http.port={self._api_port()}",
                "--assume-yes",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        # Ensure the service is enabled so it returns after a reboot.
        self._snap(wanted).start(enable=True)

    # -- Networking --------------------------------------------------------

    def _reconcile_ports(self) -> None:
        """Open the configured API port and close any stale ones."""
        wanted_port = self._api_port()
        desired = {ops.Port("tcp", wanted_port)}
        for opened in self.unit.opened_ports():
            if opened not in desired:
                self.unit.close_port(opened.protocol, opened.port)
        self.unit.open_port("tcp", wanted_port)

    def _api_url(self, wanted: str) -> str:
        """Best-effort discovery of the snap's OpenAI API URL.

        The base path of the URL depends on the active engine, so we try to
        read it from ``<snap> status``; otherwise we fall back to a plain
        host:port URL.

        ``<snap> status`` advertises several endpoints, e.g.::

            kserve: http://localhost:8080/v2
            openai: http://localhost:8080/v3
            tensorflow-serving: http://localhost:8080/v1

        We must select the ``openai`` endpoint specifically -- naively grabbing
        the first URL in the output yields the wrong base path (e.g. the kserve
        ``/v2`` endpoint) and breaks OpenAI-compatible consumers.
        """
        host = self._advertised_address()
        port = self._api_port()
        base = f"http://{host}:{port}"
        try:
            out = subprocess.run(
                [wanted, "status"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return base

        # Prefer the explicitly-labelled OpenAI endpoint.
        match = re.search(r"(?im)^\s*openai\s*:\s*http://\S*?(/\S*)", out)
        if match:
            return f"{base}{match.group(1).rstrip('/')}"
        # Fall back to the first advertised URL path if no openai label exists.
        match = re.search(r"http://[^\s]*?(/[^\s]*)", out)
        if match:
            return f"{base}{match.group(1).rstrip('/')}"
        return base

    def _advertised_address(self) -> str:
        """Address other units/machines should use to reach this unit."""
        binding = self.model.get_binding(API_RELATION)
        if binding and binding.network.ingress_address:
            return str(binding.network.ingress_address)
        return socket.getfqdn()

    def _publish_endpoint(self, wanted: str) -> None:
        """Write the endpoint details to the inference-api relation data.

        Every unit publishes its *own* endpoint to its unit databag so that a
        consumer (e.g. the model-router) can load-balance across all units of a
        scaled inference application. The leader additionally publishes
        application-level metadata for backward compatibility with consumers
        that only read the application databag.
        """
        relations = self.model.relations.get(API_RELATION, [])
        if not relations:
            return
        unit_data = {
            "snap": wanted,
            "model": wanted,
            "port": str(self._api_port()),
            "url": self._api_url(wanted),
        }
        for relation in relations:
            relation.data[self.unit].update(unit_data)
            if self.unit.is_leader():
                relation.data[self.app].update(unit_data)


if __name__ == "__main__":  # pragma: nocover
    ops.main(InferenceSnapCharm)
