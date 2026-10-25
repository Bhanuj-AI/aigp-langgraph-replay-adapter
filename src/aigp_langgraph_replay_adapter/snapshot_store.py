"""Runtime-owned private snapshot storage for the LangGraph adapter.

This module is deliberately outside AIGP OSS. It stores the private snapshot
contents needed to replay, while OSS receives only its opaque reference and
bounded integrity attestation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from bhanuj_governance_plugin_api import RuntimeReplaySource


class PrivateSnapshotError(ValueError):
    """The external runtime cannot safely resolve an isolated replay source."""


@dataclass(frozen=True)
class PrivateSnapshot:
    """Snapshot material retained solely by the external runtime."""

    replay_reference: str
    organization_id: str
    project_id: str
    external_execution_id: str
    adapter_id: str
    adapter_version: str
    isolation_guarantee: str
    manifest_hash: str
    state: Mapping[str, Any]
    snapshot_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", MappingProxyType(dict(self.state)))


class SQLitePrivateSnapshotStore:
    """Small private store used by the external runtime and local lab.

    The database is never opened by AIGP OSS. The adapter verifies every
    control-plane binding again before it exposes a snapshot to LangGraph.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def initialise(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS private_replay_snapshots (
                    replay_reference TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    external_execution_id TEXT NOT NULL,
                    adapter_id TEXT NOT NULL,
                    adapter_version TEXT NOT NULL,
                    isolation_guarantee TEXT NOT NULL,
                    manifest_hash TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL
                )
                """
            )

    def capture(
        self,
        *,
        replay_reference: str,
        organization_id: str,
        project_id: str,
        external_execution_id: str,
        adapter_id: str,
        adapter_version: str,
        isolation_guarantee: str,
        manifest_hash: str,
        state: Mapping[str, Any],
    ) -> PrivateSnapshot:
        """Store an immutable private copy, replacing only identical identity."""
        self.initialise()
        snapshot_json = _canonical_json(state)
        digest = _digest(snapshot_json)
        snapshot = PrivateSnapshot(
            replay_reference,
            organization_id,
            project_id,
            external_execution_id,
            adapter_id,
            adapter_version,
            isolation_guarantee,
            manifest_hash,
            state,
            digest,
        )
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT organization_id, project_id, external_execution_id,
                       adapter_id, adapter_version, isolation_guarantee,
                       manifest_hash, snapshot_digest
                FROM private_replay_snapshots
                WHERE replay_reference = ?
                """,
                (replay_reference,),
            ).fetchone()
            if existing is not None:
                existing_binding = tuple(str(value) for value in existing[:7])
                requested_binding = (
                    organization_id,
                    project_id,
                    external_execution_id,
                    adapter_id,
                    adapter_version,
                    isolation_guarantee,
                    manifest_hash,
                )
                if existing_binding != requested_binding or existing[7] != digest:
                    raise PrivateSnapshotError(
                        "Replay reference already identifies different private replay evidence."
                    )
            connection.execute(
                """
                INSERT OR REPLACE INTO private_replay_snapshots (
                    replay_reference, organization_id, project_id,
                    external_execution_id, adapter_id, adapter_version,
                    isolation_guarantee, manifest_hash, snapshot_json, snapshot_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    replay_reference,
                    organization_id,
                    project_id,
                    external_execution_id,
                    adapter_id,
                    adapter_version,
                    isolation_guarantee,
                    manifest_hash,
                    snapshot_json,
                    digest,
                ),
            )
        return snapshot

    def resolve(
        self,
        source: RuntimeReplaySource,
        organization_id: str,
        project_id: str,
    ) -> PrivateSnapshot:
        """Resolve only an exactly tenant- and manifest-bound private snapshot."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT replay_reference, organization_id, project_id,
                       external_execution_id, adapter_id, adapter_version,
                       isolation_guarantee, manifest_hash, snapshot_json, snapshot_digest
                FROM private_replay_snapshots
                WHERE replay_reference = ?
                """,
                (source.replay_reference,),
            ).fetchone()
        if row is None:
            raise PrivateSnapshotError("Private replay snapshot is unavailable.")
        try:
            state = json.loads(str(row[8]))
        except (TypeError, json.JSONDecodeError) as error:
            raise PrivateSnapshotError("Private replay snapshot is malformed.") from error
        if not isinstance(state, Mapping):
            raise PrivateSnapshotError("Private replay snapshot state is malformed.")
        snapshot = PrivateSnapshot(
            replay_reference=str(row[0]),
            organization_id=str(row[1]),
            project_id=str(row[2]),
            external_execution_id=str(row[3]),
            adapter_id=str(row[4]),
            adapter_version=str(row[5]),
            isolation_guarantee=str(row[6]),
            manifest_hash=str(row[7]),
            state=state,
            snapshot_digest=str(row[9]),
        )
        if _digest(_canonical_json(snapshot.state)) != snapshot.snapshot_digest:
            raise PrivateSnapshotError("Private replay snapshot integrity check failed.")
        expected = (
            (snapshot.organization_id, organization_id),
            (snapshot.project_id, project_id),
            (snapshot.external_execution_id, source.external_execution_id),
            (snapshot.adapter_id, source.adapter_id),
            (snapshot.adapter_version, source.adapter_version),
            (snapshot.isolation_guarantee, source.isolation_guarantee),
            (snapshot.manifest_hash, source.manifest_hash),
        )
        if any(actual != required for actual, required in expected):
            raise PrivateSnapshotError(
                "Private replay snapshot does not match tenant, adapter, or manifest binding."
            )
        return snapshot

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()
