from __future__ import annotations

import asyncio
import hashlib

import pytest
from bhanuj_governance_plugin_api import (
    EvidenceResolutionPlan,
    ExecutionReplayCapability,
    ReplayEvidenceBinding,
    ReplayInterventionBinding,
    ReplayResolutionCompletion,
    ReplayToolCall,
    ReplayToolCallKind,
    ToolArgumentDigestAlgorithm,
    create_tool_argument_digest,
)
from langgraph.prebuilt import ToolNode

from aigp_langgraph_replay_adapter.tool_node_replay import (
    LangGraphToolCallBinding,
    LangGraphToolNodeReplayError,
    LangGraphToolNodeReplaySession,
    PrivateToolEvidence,
    PrivateToolEvidenceStore,
)

_MANIFEST = "a" * 64


def test_tool_node_returns_recorded_evidence_without_an_original_tool() -> None:
    planned = _planned("call-1", "account_lookup", {"account_id": "A-1"}, "history")
    session = _session(
        _plan(planned),
        (
            LangGraphToolCallBinding(
                "native-1", "account_lookup", "graph.root", "v1", 1
            ),
        ),
        {"history": "historical account evidence"},
    )

    messages = session.invoke(
        (_tool_call("native-1", "account_lookup", {"account_id": "A-1"}),)
    )

    assert isinstance(session.tool_node, ToolNode)
    assert [(message.content, message.tool_call_id) for message in messages] == [
        ("historical account evidence", "native-1")
    ]
    summary = session.finalize()
    assert summary.completion is ReplayResolutionCompletion.COMPLETE
    assert summary.consumed_planned_call_ids == ("call-1",)
    assert summary.remaining_planned_call_ids == ()
    assert session.resolution_summary == summary
    assert session.execution_trace is not None
    assert session.execution_trace.complete is True
    assert session.execution_trace.resolutions[0].planned_call_id == "call-1"


def test_tool_node_uses_only_the_approved_intervention_evidence() -> None:
    planned = _planned("call-1", "account_lookup", {"account_id": "A-1"}, "history")
    plan = _plan(planned)
    intervention = ReplayInterventionBinding(
        replay_manifest_hash=plan.replay_manifest_hash,
        resolution_plan_digest=plan.plan_digest,
        target_planned_call_id=planned.call_id,
        intervention_digest="sha256:" + "f" * 64,
        replacement_evidence_reference="replacement",
        replacement_response_digest=_response_digest("approved replacement evidence"),
    )
    session = _session(
        plan,
        (
            LangGraphToolCallBinding(
                "native-1", "account_lookup", "graph.root", "v1", 1
            ),
        ),
        {
            "history": "historical account evidence",
            "replacement": "approved replacement evidence",
        },
        intervention=intervention,
    )

    messages = session.invoke(
        (_tool_call("native-1", "account_lookup", {"account_id": "A-1"}),)
    )

    assert [message.content for message in messages] == [
        "approved replacement evidence"
    ]
    summary = session.finalize()
    assert summary.completion is ReplayResolutionCompletion.COMPLETE
    assert summary.consumed_planned_call_ids == ("call-1",)
    assert summary.intervention_required is True
    assert summary.intervention_applied_call_id == "call-1"


@pytest.mark.parametrize(
    ("call", "expected_completion"),
    (
        (
            {
                "id": "native-unknown",
                "name": "unplanned_tool",
                "args": {"account_id": "A-1"},
                "type": "tool_call",
            },
            ReplayResolutionCompletion.INCOMPLETE,
        ),
        (
            {
                "id": "native-1",
                "name": "account_lookup",
                "args": {"account_id": "forged"},
                "type": "tool_call",
            },
            ReplayResolutionCompletion.FAILED,
        ),
    ),
)
def test_unknown_or_mismatched_tool_calls_fail_closed(
    call: dict[str, object], expected_completion: ReplayResolutionCompletion
) -> None:
    planned = _planned("call-1", "account_lookup", {"account_id": "A-1"}, "history")
    session = _session(
        _plan(planned),
        (
            LangGraphToolCallBinding(
                "native-1", "account_lookup", "graph.root", "v1", 1
            ),
        ),
        {"history": "historical account evidence"},
    )

    with pytest.raises(LangGraphToolNodeReplayError):
        session.invoke((call,))

    summary = session.finalize()
    assert summary.completion is expected_completion
    assert summary.remaining_planned_call_ids == ("call-1",)
    assert summary.blocking_divergence_count == (
        1 if expected_completion is ReplayResolutionCompletion.FAILED else 0
    )


def test_async_parallel_independent_calls_resolve_in_one_session() -> None:
    first = _planned("call-1", "account_lookup", {"account_id": "A-1"}, "account")
    second = _planned("call-2", "risk_lookup", {"account_id": "A-1"}, "risk")
    session = _session(
        _plan(first, second),
        (
            LangGraphToolCallBinding(
                "native-1", "account_lookup", "graph.root", "v1", 1
            ),
            LangGraphToolCallBinding("native-2", "risk_lookup", "graph.root", "v1", 1),
        ),
        {"account": "recorded account", "risk": "recorded risk"},
    )

    messages = asyncio.run(
        session.ainvoke(
            (
                _tool_call("native-1", "account_lookup", {"account_id": "A-1"}),
                _tool_call("native-2", "risk_lookup", {"account_id": "A-1"}),
            )
        )
    )

    assert {message.content for message in messages} == {
        "recorded account",
        "recorded risk",
    }
    summary = session.finalize()
    assert summary.completion is ReplayResolutionCompletion.COMPLETE
    assert summary.consumed_planned_call_ids == ("call-1", "call-2")
    assert summary.remaining_planned_call_ids == ()


def test_finalization_seals_the_session() -> None:
    planned = _planned("call-1", "account_lookup", {"account_id": "A-1"}, "history")
    session = _session(
        _plan(planned),
        (
            LangGraphToolCallBinding(
                "native-1", "account_lookup", "graph.root", "v1", 1
            ),
        ),
        {"history": "historical account evidence"},
    )

    summary = session.finalize()

    assert summary.completion is ReplayResolutionCompletion.INCOMPLETE
    with pytest.raises(LangGraphToolNodeReplayError, match="already finalized"):
        session.invoke(
            (_tool_call("native-1", "account_lookup", {"account_id": "A-1"}),)
        )


def _session(
    plan: EvidenceResolutionPlan,
    bindings: tuple[LangGraphToolCallBinding, ...],
    evidence: dict[str, str],
    *,
    intervention: ReplayInterventionBinding | None = None,
) -> LangGraphToolNodeReplaySession:
    return LangGraphToolNodeReplaySession(
        plan,
        bindings=bindings,
        evidence=PrivateToolEvidenceStore(
            {
                reference: PrivateToolEvidence(content, _response_digest(content))
                for reference, content in evidence.items()
            }
        ),
        intervention=intervention,
    )


def _plan(*calls: ReplayToolCall) -> EvidenceResolutionPlan:
    return EvidenceResolutionPlan(
        organization_id="org-a",
        project_id="project-a",
        source_execution_id="execution-1",
        replay_manifest_hash=_MANIFEST,
        capability=ExecutionReplayCapability(
            supported_tool_boundaries=("langgraph.tool-node",),
            unsupported_execution_surfaces=("direct-sdk-calls",),
            argument_digest_algorithms=(ToolArgumentDigestAlgorithm.SHA256_JSON_V1,),
            max_trace_size=16,
            supports_async_tools=True,
            supports_subgraphs=False,
            supports_retries=True,
            supports_concurrent_execution=True,
            evidence_only=True,
            no_original_tool_fallback=True,
        ),
        planned_calls=calls,
        evidence_bindings=tuple(
            ReplayEvidenceBinding(
                call.call_id,
                call.evidence_reference or "",
                call.response_digest or "",
            )
            for call in calls
        ),
    )


def _planned(
    call_id: str,
    tool_symbol: str,
    arguments: dict[str, object],
    evidence_reference: str,
) -> ReplayToolCall:
    return ReplayToolCall(
        call_id=call_id,
        kind=ReplayToolCallKind.PLANNED,
        scope_id="graph.root",
        tool_symbol=tool_symbol,
        tool_schema_version="v1",
        argument_digest=create_tool_argument_digest(
            arguments, algorithm=ToolArgumentDigestAlgorithm.SHA256_JSON_V1
        ),
        occurrence=1,
        evidence_reference=evidence_reference,
        response_digest=_response_digest(
            "historical account evidence"
            if evidence_reference == "history"
            else "recorded account"
            if evidence_reference == "account"
            else "recorded risk"
        ),
    )


def _tool_call(
    call_id: str, name: str, arguments: dict[str, object]
) -> dict[str, object]:
    return {"id": call_id, "name": name, "args": arguments, "type": "tool_call"}


def _response_digest(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
