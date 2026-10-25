# AIGP LangGraph Replay Adapter

This independently owned runtime plugin implements `langgraph/v1` using only
`bhanuj-governance-plugin-api`. It does not import AIGP OSS models, services,
repositories, or workers.

The adapter resolves an opaque `RuntimeReplaySource/v1` reference from its own
private SQLite snapshot store, verifies tenant, adapter, manifest, snapshot
digest, and isolation bindings, then invokes a LangGraph graph over an
in-memory copy. It never opens or mutates the original banking database.

```sh
export AIGP_LANGGRAPH_REPLAY_SNAPSHOT_DB=/secure/runtime/langgraph-snapshots.db
pip install aigp-langgraph-replay-adapter
```

The process that runs the AIGP replay worker discovers this package through the
`bhanuj.governance.plugins` entry point and is the only OSS process that needs
access to this private database. A manifest may be replay-eligible even when
the worker does not have this plugin installed; availability is an operational
deployment concern, not historical evidence eligibility.

The adapter treats `ReplayExecutionContext.replay_id` as the runtime replay
idempotency key. Its output contains a bounded score, opaque outcome reference,
and runtime attestation. Snapshot contents never cross into AIGP.
