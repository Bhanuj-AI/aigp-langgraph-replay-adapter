"""Worker-facing adapter for the opt-in ``langgraph/v2`` ToolNode boundary."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bhanuj_governance_plugin_api import (
    EvidenceResolutionPlan,
    ReplayExecutionAttestation,
    ReplayExecutionContext,
    ReplayExecutionPathCoverage,
    ReplayExecutionResult,
    ReplayInterventionBinding,
    ReplayResolutionCompletion,
    ReplayToolBoundaryCoverage,
    RuntimeReplaySource,
)

from aigp_langgraph_replay_adapter.snapshot_store import SQLitePrivateSnapshotStore
from aigp_langgraph_replay_adapter.tool_node_replay import (
    LangGraphToolCallBinding,
    LangGraphToolNodeReplayError,
    LangGraphToolNodeReplaySession,
    PrivateToolEvidence,
    PrivateToolEvidenceStore,
)

TOOL_NODE_REPLAY_RUNTIME_ADAPTER_ID = "langgraph/v2"
_RUNTIME_ADAPTER_VERSION = "2"
_SNAPSHOT_SCHEMA_VERSION = "langgraph-tool-node-replay-snapshot/v1"


class LangGraphToolNodeReplayAdapterError(ValueError):
    """The runtime-private v2 snapshot cannot satisfy a frozen replay request."""


@dataclass(frozen=True)
class _SnapshotCall:
    planned_call_id: str
    runtime_tool_call_id: str
    tool_symbol: str
    scope_id: str
    tool_schema_version: str
    occurrence: int
    arguments: Mapping[str, object]
    dependency_call_ids: tuple[str, ...] = ()
    retry_of_call_id: str | None = None

    def binding(self) -> LangGraphToolCallBinding:
        return LangGraphToolCallBinding(
            runtime_tool_call_id=self.runtime_tool_call_id,
            tool_symbol=self.tool_symbol,
            scope_id=self.scope_id,
            tool_schema_version=self.tool_schema_version,
            occurrence=self.occurrence,
            dependency_call_ids=self.dependency_call_ids,
            retry_of_call_id=self.retry_of_call_id,
        )

    def native_tool_call(self) -> dict[str, object]:
        return {
            "id": self.runtime_tool_call_id,
            "name": self.tool_symbol,
            "args": dict(self.arguments),
            "type": "tool_call",
        }


class LangGraphToolNodeReplayAdapter:
    """Execute declared ToolNode calls from private evidence, never original tools."""

    name = TOOL_NODE_REPLAY_RUNTIME_ADAPTER_ID

    def __init__(self, snapshots: SQLitePrivateSnapshotStore) -> None:
        self._snapshots = snapshots

    def validate_configuration(self, source_execution: object, configuration: object) -> None:
        execution_adapter = getattr(configuration, "execution_adapter", None)
        if execution_adapter != self.name:
            raise LangGraphToolNodeReplayAdapterError(
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
            raise LangGraphToolNodeReplayAdapterError("Replay cancellation requested.")
        source = self._require_source(context.runtime_replay_source)
        plan = context.execution_replay_resolution_plan
        if plan is None:
            raise LangGraphToolNodeReplayAdapterError(
                "LangGraph ToolNode replay requires a frozen evidence-resolution plan."
            )
        self._require_supported_boundary(plan)
        intervention = context.controlled_evidence_intervention
        binding = _intervention_binding(plan, intervention)
        snapshot = self._snapshots.resolve(
            source, context.organization_id, context.project_id
        )
        calls, evidence, outcome_score = _snapshot_payload(snapshot.state, plan)
        target_runtime_tool_call_id = next(
            (
                call.runtime_tool_call_id
                for call in calls
                if call.planned_call_id == binding.target_planned_call_id
            ),
            None,
        )
        if target_runtime_tool_call_id is None:
            raise LangGraphToolNodeReplayAdapterError(
                "Private snapshot lacks the authorised intervention target."
            )
        session = LangGraphToolNodeReplaySession(
            plan,
            bindings=tuple(call.binding() for call in calls),
            evidence=PrivateToolEvidenceStore(evidence),
            intervention=binding,
        )
        try:
            session.invoke(tuple(call.native_tool_call() for call in calls))
            summary = session.finalize()
        except LangGraphToolNodeReplayError as error:
            raise LangGraphToolNodeReplayAdapterError(
                "LangGraph ToolNode replay failed closed."
            ) from error
        trace = session.execution_trace
        if (
            summary.completion is not ReplayResolutionCompletion.COMPLETE
            or trace is None
            or not trace.complete
        ):
            raise LangGraphToolNodeReplayAdapterError(
                "LangGraph ToolNode replay did not complete its frozen plan."
            )

        safety = {
            "schema_version": "evidence-only-replay-safety/v1",
            "capability_schema_version": "evidence-only-replay-capability/v1",
            "replay_manifest_hash": source.manifest_hash,
            "isolation_guarantee": source.isolation_guarantee,
            "intervention_digest": _intervention_digest(intervention),
            "runtime_tool_call_id": target_runtime_tool_call_id,
            "private_snapshot_integrity_verified": True,
            "unobserved_tool_call_count": 0,
            "external_side_effects_performed": False,
        }
        attestation = ReplayExecutionAttestation(
            replay_manifest_hash=plan.replay_manifest_hash,
            resolution_plan_digest=plan.plan_digest,
            intervention_digest=binding.intervention_digest,
            phase1_safety_schema_version="evidence-only-replay-safety/v1",
            phase1_safety_digest=_safety_digest(safety),
            execution_trace_digest=trace.trace_digest,
            tool_boundary_coverage=ReplayToolBoundaryCoverage(
                declared_tool_boundaries=("langgraph.tool-node",),
                exercised_tool_boundaries=("langgraph.tool-node",),
                unsupported_execution_surfaces=(),
                execution_path_coverage=(
                    ReplayExecutionPathCoverage.COMPLETE_DECLARED_BOUNDARIES
                ),
            ),
            complete=True,
            resolved_call_count=len(trace.resolutions),
            unresolved_call_count=0,
            blocking_divergence_count=0,
            non_blocking_divergence_count=0,
        )
        outcome_ref = f"{source.replay_reference}#tool-node:{context.replay_id}"
        return ReplayExecutionResult(
            execution_status="COMPLETED",
            final_state={"causal_audit_outcome_score": outcome_score},
            events=(
                {
                    "type": "LANGGRAPH_TOOL_NODE_EVIDENCE_REPLAY_COMPLETED",
                    "resolution_plan_digest": plan.plan_digest,
                    "intervention_digest": binding.intervention_digest,
                },
            ),
            artifact_refs=(outcome_ref,),
            runtime_parameters={"evidence_only": True},
            metadata={"evidence_only_replay_safety": safety},
            execution_replay_trace=trace,
            execution_replay_attestation=attestation,
        )

    @staticmethod
    def _require_supported_boundary(plan: EvidenceResolutionPlan) -> None:
        if plan.capability.supported_tool_boundaries != ("langgraph.tool-node",):
            raise LangGraphToolNodeReplayAdapterError(
                "LangGraph v2 supports only the declared ToolNode boundary."
            )

    @staticmethod
    def _require_source(source: RuntimeReplaySource | None) -> RuntimeReplaySource:
        if source is None:
            raise LangGraphToolNodeReplayAdapterError(
                "LangGraph ToolNode replay requires RuntimeReplaySource/v1."
            )
        if (
            source.adapter_id != TOOL_NODE_REPLAY_RUNTIME_ADAPTER_ID
            or source.adapter_version != _RUNTIME_ADAPTER_VERSION
            or source.runtime_type != "langgraph"
        ):
            raise LangGraphToolNodeReplayAdapterError(
                "Runtime replay source does not identify LangGraph ToolNode v2."
            )
        return source


def _intervention_binding(
    plan: EvidenceResolutionPlan, intervention: object | None
) -> ReplayInterventionBinding:
    target = getattr(intervention, "target_event_id", None)
    reference = getattr(intervention, "counterfactual_evidence_reference", None)
    digest = getattr(intervention, "counterfactual_evidence_digest", None)
    intervention_digest = getattr(intervention, "intervention_digest", None)
    if not all(isinstance(value, str) and value for value in (target, reference, digest, intervention_digest)):
        raise LangGraphToolNodeReplayAdapterError(
            "LangGraph ToolNode replay requires a complete controlled intervention."
        )
    return ReplayInterventionBinding(
        replay_manifest_hash=plan.replay_manifest_hash,
        resolution_plan_digest=plan.plan_digest,
        target_planned_call_id=target,
        intervention_digest=f"sha256:{intervention_digest}",
        replacement_evidence_reference=reference,
        replacement_response_digest=digest,
    )


def _snapshot_payload(
    state: Mapping[str, Any], plan: EvidenceResolutionPlan
) -> tuple[tuple[_SnapshotCall, ...], dict[str, PrivateToolEvidence], float]:
    raw = state.get("tool_node_replay")
    if not isinstance(raw, Mapping) or raw.get("schema_version") != _SNAPSHOT_SCHEMA_VERSION:
        raise LangGraphToolNodeReplayAdapterError(
            "Private snapshot lacks a supported ToolNode replay payload."
        )
    calls = _snapshot_calls(raw.get("tool_calls"))
    if {call.planned_call_id for call in calls} != {
        call.call_id for call in plan.planned_calls
    }:
        raise LangGraphToolNodeReplayAdapterError(
            "Private snapshot calls do not match the frozen evidence plan."
        )
    evidence = _private_evidence(raw.get("evidence"))
    score = _outcome_score(raw.get("intervention_outcome_score"))
    return calls, evidence, score


def _snapshot_calls(value: object) -> tuple[_SnapshotCall, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LangGraphToolNodeReplayAdapterError("Private snapshot calls are invalid.")
    calls: list[_SnapshotCall] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise LangGraphToolNodeReplayAdapterError("Private snapshot call is invalid.")
        dependencies = item.get("dependency_call_ids", ())
        if not isinstance(dependencies, Sequence) or isinstance(
            dependencies, (str, bytes)
        ):
            raise LangGraphToolNodeReplayAdapterError(
                "Private snapshot call dependencies are invalid."
            )
        arguments = item.get("arguments")
        if not isinstance(arguments, Mapping):
            raise LangGraphToolNodeReplayAdapterError(
                "Private snapshot call arguments are unavailable."
            )
        try:
            calls.append(
                _SnapshotCall(
                    planned_call_id=_required_string(item, "planned_call_id"),
                    runtime_tool_call_id=_required_string(
                        item, "runtime_tool_call_id"
                    ),
                    tool_symbol=_required_string(item, "tool_symbol"),
                    scope_id=_required_string(item, "scope_id"),
                    tool_schema_version=_required_string(item, "tool_schema_version"),
                    occurrence=_required_int(item, "occurrence"),
                    arguments=dict(arguments),
                    dependency_call_ids=tuple(
                        _required_string({"value": dependency}, "value")
                        for dependency in dependencies
                    ),
                    retry_of_call_id=(
                        _required_string(item, "retry_of_call_id")
                        if item.get("retry_of_call_id") is not None
                        else None
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise LangGraphToolNodeReplayAdapterError(
                "Private snapshot call is invalid."
            ) from error
    if not calls or len({call.planned_call_id for call in calls}) != len(calls):
        raise LangGraphToolNodeReplayAdapterError(
            "Private snapshot planned call identities must be unique."
        )
    return tuple(calls)


def _private_evidence(value: object) -> dict[str, PrivateToolEvidence]:
    if not isinstance(value, Mapping):
        raise LangGraphToolNodeReplayAdapterError("Private snapshot evidence is invalid.")
    evidence: dict[str, PrivateToolEvidence] = {}
    for reference, item in value.items():
        if not isinstance(reference, str) or not isinstance(item, Mapping):
            raise LangGraphToolNodeReplayAdapterError("Private snapshot evidence is invalid.")
        try:
            evidence[reference] = PrivateToolEvidence(
                content=_required_string(item, "content"),
                response_digest=_required_string(item, "response_digest"),
            )
        except (TypeError, ValueError) as error:
            raise LangGraphToolNodeReplayAdapterError(
                "Private snapshot evidence is invalid."
            ) from error
    return evidence


def _outcome_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LangGraphToolNodeReplayAdapterError(
            "Private snapshot lacks a numeric intervention outcome."
        )
    score = float(value)
    if not math.isfinite(score):
        raise LangGraphToolNodeReplayAdapterError(
            "Private snapshot intervention outcome is not finite."
        )
    return score


def _required_string(value: Mapping[str, object], name: str) -> str:
    raw = value.get(name)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} must be a non-blank string.")
    return raw


def _required_int(value: Mapping[str, object], name: str) -> int:
    raw = value.get(name)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{name} must be an integer.")
    return raw


def _intervention_digest(intervention: object | None) -> str:
    value = getattr(intervention, "intervention_digest", None)
    if not isinstance(value, str) or not value:
        raise LangGraphToolNodeReplayAdapterError("Controlled intervention digest is missing.")
    return value


def _safety_digest(safety: Mapping[str, object]) -> str:
    payload = json.dumps(safety, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
