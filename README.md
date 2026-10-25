# AIGP LangGraph Replay Adapter

## Scope

`v0.1.0` is a **reference counterfactual executor**, not an arbitrary
LangGraph workflow replayer. It resolves a runtime-private snapshot containing
precomputed `baseline_score` and `counterfactual_scores`, then runs a small,
isolated LangGraph graph to materialize the bounded counterfactual result.
It does not load LangGraph checkpoints, reconstruct arbitrary graphs, invoke
tools, or mutate external business systems.

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
`bhanuj-governance-plugin-api==1.1.1`.

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

Snapshot capture is immutable: the first writer for a replay reference wins;
an identical later capture is idempotent, while a conflicting capture is
rejected. The adapter requires the exact `langgraph/v1` adapter id and adapter
version `1`, validates tenant, manifest, isolation and snapshot digest bindings,
and then produces a bounded score, opaque outcome reference, and runtime
attestation. The digest and attestation establish runtime consistency, not
independent proof that an external mutation occurred. Snapshot contents never
cross into AIGP.
