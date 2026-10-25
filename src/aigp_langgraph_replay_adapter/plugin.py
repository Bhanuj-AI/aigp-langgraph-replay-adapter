"""Plugin registration owned by the LangGraph runtime distribution."""

from __future__ import annotations

import os
from pathlib import Path

from bhanuj_governance_plugin_api import (
    PluginMetadata,
    ReplayExecutionAdapterContribution,
)

from aigp_langgraph_replay_adapter.adapter import LangGraphReplayAdapter
from aigp_langgraph_replay_adapter.snapshot_store import SQLitePrivateSnapshotStore


class LangGraphReplayPlugin:
    """Expose the adapter through the public AIGP plugin lifecycle."""

    metadata = PluginMetadata(
        name="langgraph-replay",
        version="0.1.0",
        required_ai_governance_version=">=1.2.4,<2",
        capabilities=("replay.execute",),
    )

    def __init__(self) -> None:
        path = os.getenv("AIGP_LANGGRAPH_REPLAY_SNAPSHOT_DB", "").strip()
        self._snapshot_path = Path(path) if path else None

    def validate(self, context) -> None:
        if self._snapshot_path is None:
            raise ValueError("AIGP_LANGGRAPH_REPLAY_SNAPSHOT_DB must be configured.")

    def register(self, context) -> None:
        assert self._snapshot_path is not None
        adapter = LangGraphReplayAdapter(SQLitePrivateSnapshotStore(self._snapshot_path))
        context.contributions.replay_execution_adapters(
            (ReplayExecutionAdapterContribution("langgraph", "v1", adapter),)
        )

    def start(self, context) -> None:
        return None

    def stop(self, context) -> None:
        return None
