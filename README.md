# inference-snap

A [Juju](https://juju.is) machine charm that deploys and operates one of
Canonical's optimized [inference snaps](https://documentation.ubuntu.com/inference-snaps/).

Each inference snap packages a generative AI model together with hardware-optimized
runtimes. The snap detects the host's CPU/GPU/NPU, runs a server as a snap
service, and exposes an OpenAI-compatible HTTP API. This charm installs the
selected snap, configures its API host/port, opens the port, and advertises the
endpoint to client charms over a relation.

## Configuration

| Option           | Type    | Default          | Description                                                                 |
| ---------------- | ------- | ---------------- | --------------------------------------------------------------------------- |
| `inference-snap` | string  | `gemma3`         | Inference snap to run. Must be a known Canonical inference snap (see below). |
| `snap-channel`   | string  | `latest/stable`  | Snap channel to install from, e.g. `latest/beta`.                           |
| `api-port`       | int     | `8080`           | Port the OpenAI-compatible API binds to (`http.port`).                      |
| `api-bind-all`   | boolean | `true`           | Bind to `0.0.0.0` (reachable on the network) vs `127.0.0.1` (local only).   |

### Supported inference snaps

`gemma3` (default), `gemma4`, `deepseek-r1`, `nemotron-3-nano`,
`nemotron-3-nano-omni`, `qwen-vl`, `qwen3`, `qwen3-coder`, `qwen3-6`.

> Note: there is no snap named exactly `gemma`; the Gemma inference snaps are
> `gemma3` and `gemma4`. The charm therefore defaults to `gemma3`.

An unrecognized `inference-snap` value puts the unit into a blocked state rather
than attempting an install.

## Usage

```bash
# Build
charmcraft pack

# Deploy with the default (gemma3)
juju deploy ./inference-snap_*.charm

# Or pick another model / channel
juju deploy ./inference-snap_*.charm \
  --config inference-snap=qwen3-coder \
  --config snap-channel=latest/beta

# Reconfigure a running unit
juju config inference-snap inference-snap=qwen3 api-port=9090
```

> Some inference snaps are currently published only to the `beta` channel. If an
> install fails on `latest/stable`, set `snap-channel=latest/beta`.

## Consuming the API from another charm

The charm provides the `inference-api` relation (interface `inference_openai`).
On relation, the provider publishes the following on its **application** data bag:

| Key     | Description                                              |
| ------- | ------------------------------------------------------- |
| `url`   | Base OpenAI-compatible API URL (incl. engine base path) |
| `port`  | API port                                                |
| `model` | Model/snap name to send in the `model` request field    |
| `snap`  | Installed inference snap name                            |

```bash
juju integrate inference-snap your-client-app
```

A client can then call, for example, `POST <url>/chat/completions` with the
advertised `model`.

The same data is published on **each unit's** relation databag (using that
unit's own address), in addition to the leader's application databag. This lets
a consumer such as the `model-router` charm load-balance across every unit of a
scaled inference application (`juju add-unit inference-snap`), not just the
leader.

## Testing

```bash
tox -e lint
tox -e unit
tox -e integration   # requires a configured Juju controller
```
