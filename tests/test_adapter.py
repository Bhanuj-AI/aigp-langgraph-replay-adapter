from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from bhanuj_governance_plugin_api import (
    EvidenceResolutionPlan,
    ExecutionReplayCapability,
    ReplayEvidenceBinding,
    ReplayExecutionContext,
    ReplayInterventionEnvelope,
    ReplayToolCall,
    ReplayToolCallKind,
    RuntimeReplaySource,
    ToolArgumentDigestAlgorithm,
    create_tool_argument_digest,
)

import aigp_langgraph_replay_adapter as public_api
from aigp_langgraph_replay_adapter.adapter import (
    LangGraphReplayAdapter,
    LangGraphReplayError,
)
from aigp_langgraph_replay_adapter.plugin import LangGraphReplayPlugin
from aigp_langgraph_replay_adapter.snapshot_store import (
    PrivateSnapshotError,
    SQLitePrivateSnapshotStore,
)
from aigp_langgraph_replay_adapter.tool_node_adapter import (
    LangGraphToolNodeReplayAdapter,
)


class _NotCancelled:
    is_cancelled = False


@dataclass(frozen=True)
class _Configuration:
    execution_adapter: str = "langgraph/v1"


class _Contributions:
    def __init__(self) -> None:
        self.items = ()

    def replay_execution_adapters(self, items) -> None:
        self.items = tuple(items)


@dataclass(frozen=True)
class _PluginContext:
    contributions: _Contributions


def test_plugin_registers_the_versioned_adapter_without_oss_imports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AIGP_LANGGRAPH_REPLAY_SNAPSHOT_DB", str(tmp_path / "private.db"))
    plugin = LangGraphReplayPlugin()
    contributions = _Contributions()

    plugin.validate(_PluginContext(contributions))
    plugin.register(_PluginContext(contributions))

    assert [
        (item.adapter_id, item.adapter_version, item.adapter.name)
        for item in contributions.items
    ] == [
        ("langgraph", "v1", "langgraph/v1"),
        ("langgraph", "v2", "langgraph/v2"),
    ]


def test_public_exports_preserve_v1_and_add_tool_node_v2() -> None:
    expected = {
        "LangGraphReplayAdapter",
        "LangGraphToolCallBinding",
        "LangGraphToolNodeReplayAdapter",
        "LangGraphToolNodeReplayAdapterError",
        "LangGraphToolNodeReplayError",
        "LangGraphToolNodeReplaySession",
        "PrivateSnapshot",
        "PrivateToolEvidence",
        "PrivateToolEvidenceStore",
        "SQLitePrivateSnapshotStore",
        "TOOL_NODE_REPLAY_ADAPTER_ID",
        "TOOL_NODE_REPLAY_RUNTIME_ADAPTER_ID",
    }

    assert set(public_api.__all__) == expected
    assert all(hasattr(public_api, name) for name in expected)


def test_v2_adapter_returns_trace_and_attestation_from_private_evidence(
    tmp_path: Path,
) -> None:
    source = _source(adapter_id="langgraph/v2", adapter_version="2")
    arguments = {"account_id": "A-1"}
    history = "recorded account evidence"
    replacement = "approved replacement evidence"
    plan = _v2_plan(arguments, _digest(history))
    snapshots = SQLitePrivateSnapshotStore(tmp_path / "private-snapshots.db")
    snapshots.capture(
        replay_reference=source.replay_reference,
        organization_id="org-a",
        project_id="project-a",
        external_execution_id=source.external_execution_id,
        adapter_id=source.adapter_id,
        adapter_version=source.adapter_version,
        isolation_guarantee=source.isolation_guarantee,
        manifest_hash=source.manifest_hash,
        state={
            "tool_node_replay": {
                "schema_version": "langgraph-tool-node-replay-snapshot/v1",
                "tool_calls": [
                    {
                        "planned_call_id": "tool-call-1",
                        "runtime_tool_call_id": "native-account-1",
                        "tool_symbol": "account_lookup",
                        "scope_id": "graph.root",
                        "tool_schema_version": "v1",
                        "occurrence": 1,
                        "dependency_call_ids": [],
                        "arguments": arguments,
                    }
                ],
                "evidence": {
                    "private://history": {
                        "content": history,
                        "response_digest": _digest(history),
                    },
                    "private://replacement": {
                        "content": replacement,
                        "response_digest": _digest(replacement),
                    },
                },
                "intervention_outcome_score": 0.2,
            }
        },
    )
    intervention = SimpleNamespace(
        target_event_id="tool-call-1",
        counterfactual_evidence_reference="private://replacement",
        counterfactual_evidence_digest=_digest(replacement),
        intervention_digest="f" * 64,
    )
    context = ReplayExecutionContext(
        replay_id="replay-1",
        source_execution_id="source-1",
        new_execution_id="produced-1",
        organization_id="org-a",
        project_id="project-a",
        actor_id="auditor",
        request_id="request-1",
        correlation_id="correlation-1",
        attempt=1,
        cancellation_token=_NotCancelled(),
        metadata={},
        controlled_evidence_intervention=intervention,
        runtime_replay_source=source,
        execution_replay_resolution_plan=plan,
    )

    result = LangGraphToolNodeReplayAdapter(snapshots).replay(
        object(),
        SimpleNamespace(execution_adapter="langgraph/v2"),
        context,
    )

    assert result.final_state["causal_audit_outcome_score"] == 0.2
    assert result.execution_replay_trace is not None
    assert result.execution_replay_trace.complete is True
    assert result.execution_replay_attestation is not None
    assert result.execution_replay_attestation.complete is True
    assert result.metadata["evidence_only_replay_safety"][
        "runtime_tool_call_id"
    ] == "native-account-1"


def test_plugin_without_worker_snapshot_configuration_remains_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AIGP_LANGGRAPH_REPLAY_SNAPSHOT_DB", raising=False)
    plugin = LangGraphReplayPlugin()
    contributions = _Contributions()

    plugin.validate(_PluginContext(contributions))
    plugin.register(_PluginContext(contributions))

    assert contributions.items == ()


def test_isolated_langgraph_replay_preserves_original_banking_database(
    tmp_path: Path,
) -> None:
    business_database = tmp_path / "banking.db"
    with sqlite3.connect(business_database) as connection:
        connection.execute(
            "CREATE TABLE loan_applications (loan_id TEXT PRIMARY KEY, status TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO loan_applications VALUES ('LOAN-001', 'PENDING')"
        )
    source = _source()
    snapshots = SQLitePrivateSnapshotStore(tmp_path / "private-snapshots.db")
    snapshots.capture(
        replay_reference=source.replay_reference,
        organization_id="org-a",
        project_id="project-a",
        external_execution_id=source.external_execution_id,
        adapter_id=source.adapter_id,
        adapter_version=source.adapter_version,
        isolation_guarantee=source.isolation_guarantee,
        manifest_hash=source.manifest_hash,
        state={
            "baseline_score": 1.0,
            "counterfactual_scores": {"sha256:counterfactual": 0.2},
        },
    )

    result = LangGraphReplayAdapter(snapshots).replay(
        object(), _Configuration(), _context(source)
    )

    with sqlite3.connect(business_database) as connection:
        status = connection.execute(
            "SELECT status FROM loan_applications WHERE loan_id = 'LOAN-001'"
        ).fetchone()[0]
    assert status == "PENDING"
    assert result.final_state["causal_audit_outcome_score"] == 0.2
    assert result.input == {}
    assert result.runtime_parameters == {"isolated": True}
    assert result.metadata["runtime_replay_attestation"][
        "private_snapshot_integrity_verified"
    ] is True
    assert result.metadata["runtime_replay_attestation"]["adapter_version"] == "1"
    assert result.metadata["evidence_only_replay_safety"] == {
        "schema_version": "evidence-only-replay-safety/v1",
        "capability_schema_version": "evidence-only-replay-capability/v1",
        "replay_manifest_hash": source.manifest_hash,
        "isolation_guarantee": source.isolation_guarantee,
        "intervention_digest": "sha256:intervention",
        "runtime_tool_call_id": "approve-loan-1",
        "private_snapshot_integrity_verified": True,
        "unobserved_tool_call_count": 0,
        "external_side_effects_performed": False,
    }


def test_private_snapshot_resolution_rejects_cross_tenant_and_binding_forgery(
    tmp_path: Path,
) -> None:
    source = _source()
    snapshots = SQLitePrivateSnapshotStore(tmp_path / "private-snapshots.db")
    snapshots.capture(
        replay_reference=source.replay_reference,
        organization_id="org-a",
        project_id="project-a",
        external_execution_id=source.external_execution_id,
        adapter_id=source.adapter_id,
        adapter_version=source.adapter_version,
        isolation_guarantee=source.isolation_guarantee,
        manifest_hash=source.manifest_hash,
        state={
            "baseline_score": 1.0,
            "counterfactual_scores": {"sha256:counterfactual": 0.2},
        },
    )

    with pytest.raises(PrivateSnapshotError, match="tenant"):
        snapshots.resolve(source, "org-b", "project-a")
    with pytest.raises(PrivateSnapshotError, match="tenant, adapter, or manifest"):
        snapshots.resolve(
            RuntimeReplaySource(
                source_agent_execution_id=source.source_agent_execution_id,
                external_execution_id=source.external_execution_id,
                runtime_type=source.runtime_type,
                adapter_id=source.adapter_id,
                adapter_version=source.adapter_version,
                replay_reference=source.replay_reference,
                isolation_guarantee=source.isolation_guarantee,
                manifest_hash="b" * 64,
            ),
            "org-a",
            "project-a",
        )


def test_adapter_rejects_missing_runtime_replay_source(tmp_path: Path) -> None:
    adapter = LangGraphReplayAdapter(SQLitePrivateSnapshotStore(tmp_path / "private.db"))
    context = ReplayExecutionContext(
        replay_id="replay-1",
        source_execution_id="source-1",
        new_execution_id="produced-1",
        organization_id="org-a",
        project_id="project-a",
        actor_id="auditor",
        request_id="request-1",
        correlation_id="correlation-1",
        attempt=1,
        cancellation_token=_NotCancelled(),
        metadata={},
        intervention_envelope=_envelope(),
    )

    with pytest.raises(LangGraphReplayError, match="RuntimeReplaySource"):
        adapter.replay(object(), _Configuration(), context)


@pytest.mark.parametrize(
    ("adapter_id", "adapter_version"),
    (("forged/langgraph", "1"), ("langgraph/v1", "2")),
)
def test_adapter_rejects_forged_adapter_identity_or_unsupported_version(
    tmp_path: Path, adapter_id: str, adapter_version: str
) -> None:
    source = _source(adapter_id=adapter_id, adapter_version=adapter_version)
    adapter = LangGraphReplayAdapter(SQLitePrivateSnapshotStore(tmp_path / "private.db"))

    with pytest.raises(LangGraphReplayError, match="supported LangGraph adapter"):
        adapter.replay(object(), _Configuration(), _context(source))


def test_snapshot_capture_allows_idempotent_concurrent_duplicates(tmp_path: Path) -> None:
    source = _source()
    store = SQLitePrivateSnapshotStore(tmp_path / "private-snapshots.db")
    store.initialise()
    barrier = threading.Barrier(2)

    def capture() -> str:
        barrier.wait()
        return _capture(store, source, 0.2).snapshot_digest

    with ThreadPoolExecutor(max_workers=2) as executor:
        digests = list(executor.map(lambda _: capture(), range(2)))

    assert digests == [digests[0], digests[0]]
    assert store.resolve(source, "org-a", "project-a").snapshot_digest == digests[0]


def test_snapshot_capture_rejects_concurrent_conflicts_without_replacement(
    tmp_path: Path,
) -> None:
    source = _source()
    store = SQLitePrivateSnapshotStore(tmp_path / "private-snapshots.db")
    store.initialise()
    barrier = threading.Barrier(2)

    def capture(score: float) -> str | PrivateSnapshotError:
        barrier.wait()
        try:
            return _capture(store, source, score).snapshot_digest
        except PrivateSnapshotError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(capture, (0.2, 0.7)))

    successes = [result for result in results if isinstance(result, str)]
    conflicts = [result for result in results if isinstance(result, PrivateSnapshotError)]
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert "different private replay evidence" in str(conflicts[0])
    assert store.resolve(source, "org-a", "project-a").snapshot_digest == successes[0]


def _source(
    adapter_id: str = "langgraph/v1", adapter_version: str = "1"
) -> RuntimeReplaySource:
    return RuntimeReplaySource(
        source_agent_execution_id="aigp-execution-1",
        external_execution_id="langgraph-run-1",
        runtime_type="langgraph",
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        replay_reference="langgraph://replays/langgraph-run-1",
        isolation_guarantee="sqlite-copy/v1",
        manifest_hash="a" * 64,
    )


def _context(source: RuntimeReplaySource) -> ReplayExecutionContext:
    return ReplayExecutionContext(
        replay_id="replay-1",
        source_execution_id="source-1",
        new_execution_id="produced-1",
        organization_id="org-a",
        project_id="project-a",
        actor_id="auditor",
        request_id="request-1",
        correlation_id="correlation-1",
        attempt=1,
        cancellation_token=_NotCancelled(),
        metadata={},
        runtime_replay_source=source,
        intervention_envelope=_envelope(),
    )


def _envelope() -> ReplayInterventionEnvelope:
    return ReplayInterventionEnvelope(
        policy_id="policy-1",
        policy_version=1,
        external_execution_id="langgraph-run-1",
        runtime_tool_call_id="approve-loan-1",
        intervention_provider="opaque-reference",
        intervention_provider_version="v1",
        strategy="REPLACE",
        original_evidence_digest="sha256:original",
        counterfactual_reference="runtime://counterfactual/loan-1",
        counterfactual_digest="sha256:counterfactual",
        intervention_digest="sha256:intervention",
    )


def _capture(
    store: SQLitePrivateSnapshotStore, source: RuntimeReplaySource, score: float
):
    return store.capture(
        replay_reference=source.replay_reference,
        organization_id="org-a",
        project_id="project-a",
        external_execution_id=source.external_execution_id,
        adapter_id=source.adapter_id,
        adapter_version=source.adapter_version,
        isolation_guarantee=source.isolation_guarantee,
        manifest_hash=source.manifest_hash,
        state={
            "baseline_score": 1.0,
            "counterfactual_scores": {"sha256:counterfactual": score},
        },
    )


def _v2_plan(
    arguments: dict[str, object], response_digest: str
) -> EvidenceResolutionPlan:
    planned = ReplayToolCall(
        call_id="tool-call-1",
        kind=ReplayToolCallKind.PLANNED,
        scope_id="graph.root",
        tool_symbol="account_lookup",
        tool_schema_version="v1",
        argument_digest=create_tool_argument_digest(
            arguments, algorithm=ToolArgumentDigestAlgorithm.SHA256_JSON_V1
        ),
        occurrence=1,
        evidence_reference="private://history",
        response_digest=response_digest,
    )
    return EvidenceResolutionPlan(
        organization_id="org-a",
        project_id="project-a",
        source_execution_id="aigp-execution-1",
        replay_manifest_hash="a" * 64,
        capability=ExecutionReplayCapability(
            supported_tool_boundaries=("langgraph.tool-node",),
            unsupported_execution_surfaces=("direct-sdk-calls",),
            argument_digest_algorithms=(ToolArgumentDigestAlgorithm.SHA256_JSON_V1,),
            max_trace_size=8,
            supports_async_tools=True,
            supports_subgraphs=False,
            supports_retries=True,
            supports_concurrent_execution=True,
            evidence_only=True,
            no_original_tool_fallback=True,
        ),
        planned_calls=(planned,),
        evidence_bindings=(
            ReplayEvidenceBinding(
                planned.call_id,
                planned.evidence_reference or "",
                planned.response_digest or "",
            ),
        ),
    )


def _digest(content: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
