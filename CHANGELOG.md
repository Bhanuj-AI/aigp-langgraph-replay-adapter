# Changelog

All notable changes to this package are documented here.

## [0.3.1] - 2026-11-21

### Fixed

- Moved the development-only replay-worker image and Compose overlay into this
  adapter repository. The OSS control plane now remains free of
  framework-specific deployment files.
- Made the overlay portable through explicit absolute adapter and private
  snapshot directory variables. It layers this adapter onto a supplied AIGP
  worker image and mounts the runtime-owned snapshot directory read-only into
  the worker only.

## [0.3.0] - 2026-11-21

### Added

- Added the opt-in `langgraph/v2` replay adapter for the declared LangGraph
  `ToolNode` boundary. It builds resolver-backed tools only; original tool
  implementations are never supplied or used.
- Added a typed execution trace and Phase 2 attestation based on the frozen
  evidence-resolution plan and one authorised intervention binding.
- Added complete-session enforcement: every planned call must be consumed once,
  the intervention target must be resolved once, and incomplete or divergent
  ToolNode sessions fail closed.

### Compatibility

- Retained the existing `langgraph/v1` reference replay contribution. The v2
  adapter is separately registered and requires an observed execution to opt
  in with adapter ID `langgraph/v2` and version `2`.

### Boundary

- This release resolves evidence for declared calls through `ToolNode`. It does
  not intercept custom graph nodes, direct SDK calls, provider-hosted tools, or
  arbitrary workflow code. It is not sandbox-level isolation.
