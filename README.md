# AIGP LangGraph Replay Adapter

## Scope

`v0.3.1` preserves the existing `langgraph/v1` adapter as a **reference counterfactual executor**, not an arbitrary
LangGraph workflow replayer. It resolves a runtime-private snapshot containing
precomputed `baseline_score` and `counterfactual_scores`, then runs a small,
isolated LangGraph graph to materialize the bounded counterfactual result.
It does not load LangGraph checkpoints, reconstruct arbitrary graphs, invoke
tools, or mutate external business systems.

The opt-in ToolNode replay surface is a separate `langgraph/v2` boundary. It
uses the published Plugin API resolver to substitute private recorded evidence
for declared native `ToolNode` calls. When an OSS worker receives a frozen
resolution plan and authorised intervention binding, the adapter emits the
typed execution trace, evidence-only safety report, and attestation that the
worker independently reconciles before accepting the replay result.

This independently owned runtime plugin implements `langgraph/v1` using only
`bhanuj-governance-plugin-api`. It does not import AIGP OSS models, services,
repositories, or workers.

The adapter resolves an opaque `RuntimeReplaySource/v1` reference from its own
private SQLite snapshot store, verifies tenant, adapter, manifest, snapshot
digest, and isolation bindings, then invokes a LangGraph graph over an
in-memory copy. It never opens or mutates the original banking database.

## Installation

The adapter package is not yet published to PyPI. Install it from a checked-out
release tag or a locally built wheel instead:

```sh
git clone https://github.com/Bhanuj-AI/aigp-langgraph-replay-adapter.git
cd aigp-langgraph-replay-adapter
pip install .
```

For local development, use `pip install -e .`. The published Plugin API is a
normal dependency and is resolved by the project at
`bhanuj-governance-plugin-api==1.2.0rc1`.

The replay worker must be given its private snapshot location:

```sh
export AIGP_LANGGRAPH_REPLAY_SNAPSHOT_DB=/secure/runtime/langgraph-snapshots.db
```

The process that runs the AIGP replay worker discovers this package through the
`bhanuj.governance.plugins` entry point and is the only OSS process that needs
access to this private database. If the package is visible in an API process
but `AIGP_LANGGRAPH_REPLAY_SNAPSHOT_DB` is absent, discovery leaves it inert:
no replay adapter is registered. A governed replay request consequently fails
closed as unavailable rather than silently falling back. A manifest may still
be replay-eligible when the current worker lacks the adapter; availability is
an operational deployment concern, not historical evidence eligibility.

### Local Docker worker

This repository owns the development-only image and Compose overlay for the
external adapter. The OSS repository remains framework-neutral. The overlay
layers this package onto an already-built AIGP worker image and mounts the
runtime-owned private snapshot directory read-only into that worker only.

```sh
# From this adapter checkout.
export AIGP_LANGGRAPH_REPLAY_ADAPTER_DIR="$(pwd)"
export AIGP_LANGGRAPH_REPLAY_SNAPSHOT_DIR="/absolute/path/to/runtime-data"
export AIGP_BASE_WORKER_IMAGE=ai-governance-platform:local
export AIGP_LANGGRAPH_REPLAY_WORKER_IMAGE=ai-governance-platform-langgraph-v2:local

# Build the base control-plane image from its own checkout first.
cd ../ai-governance-oss
docker compose build ai-governance-platform
docker compose -f docker-compose.yml \
  -f ../adapters/aigp-langgraph-replay-adapter/docker-compose.replay-worker.yml \
  build ai-governance-replay-worker
docker compose -f docker-compose.yml \
  -f ../adapters/aigp-langgraph-replay-adapter/docker-compose.replay-worker.yml \
  up -d --force-recreate ai-governance-platform ai-governance-replay-worker
```

The API and Studio do not receive the snapshot mount. This is a deployment
boundary for the declared ToolNode replay surface; it is not a claim of
arbitrary workflow interception or sandbox-level isolation.

Snapshot capture is immutable: the first writer for a replay reference wins;
an identical later capture is idempotent, while a conflicting capture is
rejected. The adapter requires the exact `langgraph/v1` adapter id and adapter
version `1`, validates tenant, manifest, isolation and snapshot digest bindings,
and then produces a bounded score, opaque outcome reference, and runtime
attestation. The digest and attestation establish runtime consistency, not
independent proof that an external mutation occurred. Snapshot contents never
cross into AIGP.

## Resolver-backed ToolNode replay (`langgraph/v2`)

`LangGraphToolNodeReplaySession` is the first concrete interception boundary.
It creates a real LangGraph `ToolNode` containing only resolver-backed tools;
the original tool implementations are never accepted or invoked. The runtime
privately supplies the native call ID, scope, schema, occurrence, dependencies,
retry relationship, and a private evidence materialiser. The session derives a
canonical argument digest, invokes `ExecutionReplayResolver`, and returns only
the selected recorded or approved-replacement evidence to `ToolNode`.

Independent calls may resolve in parallel through `ToolNode.ainvoke()` because
the published resolver atomically matches and consumes each planned call. An
unknown ToolNode name or an identity/digest mismatch terminates the attempted
path without falling back to the original tool. Direct SDK, HTTP, database, and
custom-node calls remain outside this declared boundary and are not represented
as intercepted execution.

The runtime must call `finalize()` before considering the session successful.
It returns and retains the immutable `ReplayResolutionSummary`, including its
completion state, consumed and remaining planned calls, blocking divergence
count, and whether a required intervention was applied. Finalization seals the
session: no further ToolNode invocation is allowed. A later OSS worker
integration must validate this summary before accepting a replay result.
