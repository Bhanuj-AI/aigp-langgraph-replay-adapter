"""Resolver-backed LangGraph ``ToolNode`` replay surface.

This is an opt-in ``langgraph/v2`` boundary. It deliberately creates only
resolver-backed tools; callers never supply the original tool implementations.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any

from bhanuj_governance_plugin_api import (
    EvidenceResolutionPlan,
    ExecutionReplayResolver,
    ReplayInterventionBinding,
    ReplayResolutionCompletion,
    ReplayResolutionOutcome,
    ReplayToolCall,
    ReplayToolCallKind,
    ToolArgumentDigestAlgorithm,
    ToolArgumentDigestKeyProvider,
    create_tool_argument_digest,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict

TOOL_NODE_REPLAY_ADAPTER_ID = "langgraph/v2"


class LangGraphToolNodeReplayError(ValueError):
    """A ToolNode call cannot safely be replayed from recorded evidence."""


@dataclass(frozen=True)
class LangGraphToolCallBinding:
    """Runtime-private identity data for one native LangGraph tool call."""

    runtime_tool_call_id: str
    tool_symbol: str
    scope_id: str
    tool_schema_version: str
    occurrence: int
    dependency_call_ids: tuple[str, ...] = ()
    retry_of_call_id: str | None = None
    argument_digest_algorithm: ToolArgumentDigestAlgorithm = (
        ToolArgumentDigestAlgorithm.SHA256_JSON_V1
    )
    argument_digest_key_version: str | None = None


@dataclass(frozen=True)
class PrivateToolEvidence:
    """Runtime-private tool response material; never returned to AIGP OSS."""

    content: str
    response_digest: str


class PrivateToolEvidenceStore:
    """In-memory private evidence materialiser for one ToolNode replay session."""

    def __init__(self, evidence: Mapping[str, PrivateToolEvidence]) -> None:
        self._evidence = MappingProxyType(dict(evidence))

    def materialize(self, reference: str, expected_digest: str) -> str:
        item = self._evidence.get(reference)
        if item is None:
            raise LangGraphToolNodeReplayError(
                "Recorded private tool evidence is unavailable."
            )
        actual_digest = (
            "sha256:" + hashlib.sha256(item.content.encode("utf-8")).hexdigest()
        )
        if item.response_digest != expected_digest or actual_digest != expected_digest:
            raise LangGraphToolNodeReplayError(
                "Recorded private tool evidence digest does not match the resolution."
            )
        return item.content


class _ResolverToolArguments(BaseModel):
    """Accept the native tool's arguments while ToolNode injects its call ID."""

    model_config = ConfigDict(extra="allow")

    tool_call_id: Annotated[str, InjectedToolCallId]


class _FailClosedToolNode(ToolNode):
    """Make an unknown ToolNode name terminal instead of emitting an error message."""

    def _validate_tool_call(self, call: Any) -> ToolMessage | None:
        invalid = super()._validate_tool_call(call)
        if invalid is not None:
            raise LangGraphToolNodeReplayError(
                f"Unobserved LangGraph tool call: {call['name']}."
            )
        return None


class LangGraphToolNodeReplaySession:
    """Run one declared ToolNode surface against immutable replay evidence.

    ``ToolNode`` supplies its native call ID to each resolver-backed tool and
    schedules independent calls normally. The OSS resolver owns the atomic
    match, dependency check, and planned-call consumption.
    """

    adapter_id = TOOL_NODE_REPLAY_ADAPTER_ID

    def __init__(
        self,
        plan: EvidenceResolutionPlan,
        *,
        bindings: Sequence[LangGraphToolCallBinding],
        evidence: PrivateToolEvidenceStore,
        intervention: ReplayInterventionBinding | None = None,
        key_provider: ToolArgumentDigestKeyProvider | None = None,
    ) -> None:
        self._plan = plan
        self._resolver = ExecutionReplayResolver(plan, intervention)
        self._evidence = evidence
        self._key_provider = key_provider
        self._intervention_digest = (
            intervention.intervention_digest if intervention is not None else None
        )
        by_runtime_id = {binding.runtime_tool_call_id: binding for binding in bindings}
        if len(by_runtime_id) != len(bindings):
            raise LangGraphToolNodeReplayError(
                "LangGraph runtime tool-call bindings must be unique."
            )
        self._bindings = MappingProxyType(by_runtime_id)
        tool_symbols = tuple(
            sorted({planned.tool_symbol for planned in self._plan.planned_calls})
        )
        self.tool_node = _FailClosedToolNode(
            [self._resolver_tool(symbol) for symbol in tool_symbols],
            handle_tool_errors=False,
        )

    def invoke(self, tool_calls: Sequence[Mapping[str, object]]) -> list[ToolMessage]:
        """Synchronously run native calls through a real LangGraph ``ToolNode``."""

        result = self.tool_node.invoke(_tool_node_input(tool_calls))
        return _tool_messages(result)

    async def ainvoke(
        self, tool_calls: Sequence[Mapping[str, object]]
    ) -> list[ToolMessage]:
        """Asynchronously run native calls; ToolNode may resolve them in parallel."""

        result = await self.tool_node.ainvoke(_tool_node_input(tool_calls))
        return _tool_messages(result)

    def finalize(self) -> ReplayResolutionCompletion:
        """Return the session completion state for runtime-owned trace handling."""

        return self._resolver.finalize().completion

    def _resolver_tool(self, tool_symbol: str):
        @tool(tool_symbol, args_schema=_ResolverToolArguments)
        def resolve_recorded_evidence(tool_call_id: str, **arguments: object) -> str:
            """Resolve one declared call to private recorded evidence."""

            return self._resolve(tool_symbol, tool_call_id, arguments)

        return resolve_recorded_evidence

    def _resolve(
        self,
        tool_symbol: str,
        runtime_tool_call_id: str,
        arguments: Mapping[str, object],
    ) -> str:
        binding = self._bindings.get(runtime_tool_call_id)
        if binding is None:
            raise LangGraphToolNodeReplayError(
                "LangGraph tool call has no runtime-private replay binding."
            )
        if binding.tool_symbol != tool_symbol:
            raise LangGraphToolNodeReplayError(
                "LangGraph tool call does not match its declared tool binding."
            )
        try:
            argument_digest = create_tool_argument_digest(
                arguments,
                algorithm=binding.argument_digest_algorithm,
                organization_id=self._plan.organization_id,
                project_id=self._plan.project_id,
                key_version=binding.argument_digest_key_version,
                key_provider=self._key_provider,
            )
            attempted = ReplayToolCall(
                call_id=runtime_tool_call_id,
                kind=ReplayToolCallKind.ATTEMPTED,
                scope_id=binding.scope_id,
                tool_symbol=tool_symbol,
                tool_schema_version=binding.tool_schema_version,
                argument_digest=argument_digest,
                occurrence=binding.occurrence,
                dependency_call_ids=binding.dependency_call_ids,
                retry_of_call_id=binding.retry_of_call_id,
            )
        except (TypeError, ValueError) as error:
            raise LangGraphToolNodeReplayError(
                "LangGraph tool call cannot be canonicalized for replay."
            ) from error
        resolution = self._resolver.resolve(
            attempted, intervention_digest=self._intervention_digest
        )
        if resolution.outcome not in {
            ReplayResolutionOutcome.RESOLVED_RECORDED,
            ReplayResolutionOutcome.RESOLVED_INTERVENTION,
        }:
            raise LangGraphToolNodeReplayError(
                f"LangGraph tool call failed closed: {resolution.reason_code}."
            )
        if (
            resolution.selected_evidence_reference is None
            or resolution.selected_response_digest is None
        ):
            raise LangGraphToolNodeReplayError(
                "Resolver selected no private evidence for a successful tool call."
            )
        return self._evidence.materialize(
            resolution.selected_evidence_reference,
            resolution.selected_response_digest,
        )


def _tool_node_input(tool_calls: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {"messages": [AIMessage(content="", tool_calls=list(tool_calls))]}


def _tool_messages(result: Mapping[str, object]) -> list[ToolMessage]:
    messages = result.get("messages")
    if not isinstance(messages, list) or not all(
        isinstance(message, ToolMessage) for message in messages
    ):
        raise LangGraphToolNodeReplayError("ToolNode did not return tool messages.")
    return messages
