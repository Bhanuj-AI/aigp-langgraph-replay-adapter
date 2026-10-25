"""Versioned adapter that runs a LangGraph replay against private copies only."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from bhanuj_governance_plugin_api import (
    ReplayExecutionContext,
    ReplayExecutionResult,
    RuntimeReplaySource,
)

from aigp_langgraph_replay_adapter.snapshot_store import (
    SQLitePrivateSnapshotStore,
)


class LangGraphReplayError(ValueError):
    """The LangGraph runtime rejected a governed replay request."""


class _ReplayState(TypedDict):
    baseline_score: float
    replay_score: float


@dataclass(frozen=True)
class LangGraphReplayAdapter:
    """Run reference-bound counterfactuals without touching business state.

    The adapter is intentionally independent of AIGP OSS. It resolves private
    snapshots through its own store, validates every control-plane binding,
    and uses a LangGraph graph over in-memory state only.
    """

    snapshots: SQLitePrivateSnapshotStore
    name: str = "langgraph/v1"

    def validate_configuration(self, source_execution: object, configuration: object) -> None:
        execution_adapter = getattr(configuration, "execution_adapter", None)
        if execution_adapter != self.name:
            raise LangGraphReplayError(
                "Frozen replay configuration selected a different runtime adapter."
            )

    def replay(
        self,
        source_execution: object,
        configuration: object,
        context: ReplayExecutionContext,
    ) -> ReplayExecutionResult:
        self.validate_configuration(source_execution, configuration)
        if context.cancellation_token.is_cancelled:
            raise LangGraphReplayError("Replay cancellation requested.")
        source = self._require_source(context.runtime_replay_source)
        envelope = context.intervention_envelope
        if envelope is None:
            raise LangGraphReplayError(
                "Governed LangGraph replay requires a controlled intervention envelope."
            )
        if envelope.external_execution_id != source.external_execution_id:
            raise LangGraphReplayError(
                "Controlled intervention does not belong to the runtime replay source."
            )
        snapshot = self.snapshots.resolve(
            source, context.organization_id, context.project_id
        )
        score = self._run_isolated_graph(snapshot.state, envelope.strategy, envelope.counterfactual_digest)
        outcome_ref = (
            f"{source.replay_reference}#replay:{context.replay_id}:"
            f"{envelope.intervention_digest}"
        )
        return ReplayExecutionResult(
            execution_status="COMPLETED",
            final_state={
                "causal_audit_outcome_score": score,
                "outcome_ref": outcome_ref,
                "private_snapshot_digest": snapshot.snapshot_digest,
            },
            events=(
                {
                    "type": "LANGGRAPH_CONTROLLED_REPLAY_COMPLETED",
                    "runtime_tool_call_id": envelope.runtime_tool_call_id,
                    "intervention_digest": envelope.intervention_digest,
                },
            ),
            artifact_refs=(outcome_ref,),
            runtime_parameters={"isolated": True},
            metadata={
                "runtime_replay_attestation": {
                    "adapter_id": self.name,
                    "manifest_hash": source.manifest_hash,
                    "isolation_guarantee": source.isolation_guarantee,
                    "private_snapshot_integrity_verified": True,
                }
            },
        )

    def _require_source(
        self, source: RuntimeReplaySource | None
    ) -> RuntimeReplaySource:
        if source is None:
            raise LangGraphReplayError(
                "Governed LangGraph replay requires RuntimeReplaySource/v1."
            )
        if source.adapter_id != self.name or source.runtime_type != "langgraph":
            raise LangGraphReplayError(
                "Runtime replay source does not identify this LangGraph adapter."
            )
        return source

    @staticmethod
    def _run_isolated_graph(
        state: Mapping[str, object], strategy: str, counterfactual_digest: str
    ) -> float:
        baseline = state.get("baseline_score")
        alternatives = state.get("counterfactual_scores")
        if not isinstance(baseline, (int, float)) or isinstance(baseline, bool):
            raise LangGraphReplayError("Private snapshot lacks a numeric baseline score.")
        if not isinstance(alternatives, Mapping):
            raise LangGraphReplayError("Private snapshot lacks counterfactual outcomes.")
        replacement = alternatives.get(counterfactual_digest)
        if strategy == "NULLIFY":
            replacement = 0.0
        if not isinstance(replacement, (int, float)) or isinstance(replacement, bool):
            raise LangGraphReplayError(
                "Private snapshot lacks a numeric outcome for this intervention."
            )
        if not math.isfinite(float(replacement)):
            raise LangGraphReplayError("Private snapshot outcome is not finite.")

        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(_ReplayState)
        graph.add_node(
            "apply_governed_intervention",
            lambda _state: {"replay_score": float(replacement)},
        )
        graph.add_edge(START, "apply_governed_intervention")
        graph.add_edge("apply_governed_intervention", END)
        result = graph.compile().invoke(
            {"baseline_score": float(baseline), "replay_score": float(baseline)}
        )
        score = result.get("replay_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise LangGraphReplayError("LangGraph replay did not produce an outcome score.")
        return float(score)
