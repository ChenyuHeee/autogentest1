# Requirements Inventory

This inventory tracks the major capability requirements captured from trader feedback
and the upgrade roadmap. Update the status column as work progresses.

| Capability | Description | Primary Artifacts | Status | Notes |
| --- | --- | --- | --- | --- |
| Retrieval-Augmented briefings | Agents can cite macro history and trading playbooks in outputs | `scripts/ingest_macro_history.py`, `autogentest1/tools/rag` | ✅ Complete | Baseline corpus ingested; monitor coverage expansion backlog. |
| Structured audit logging | Persist JSONL audit events for risk/circuit breaker decisions | `services/audit.py`, `docs/workflows/audit_logging.md` | ✅ Complete | Defaults now point to `~/.autogentest1/audit`; ensure ops rotation. |
| Parameter sweep backtesting | Grid-search helper with ranking, metrics fallback, and docs/tests | `services/backtest_suite.py`, `tools/backtest_tools.py`, `tests/test_backtest_suite.py` | ✅ Complete | Integrate results into agent prompts in a later sprint. |
| Hard risk gate enhancements | Data-quality aware limits, incremental utilization, session relaxations | `services/risk_gate.py`, `services/risk.py`, `config/settings.py` | ✅ Complete | Continue tuning thresholds with live calibration data. |
| Portfolio state persistence | Deep merge + patch application for circuit breaker state | `services/state.py`, `tests/test_state.py` | ✅ Complete | Extend to multi-asset holdings in Phase 2. |
| Requirement inventory tracking | Central document enumerating roadmap requirements | `docs/requirements/requirements_inventory.md` | ✅ Complete | Review during weekly stand-up to reprioritize backlog. |
| Vector store scalability plan | Prototype index reporting, evaluate FAISS/Chroma footprint | `docs/requirements/phase-upgrade-requirements.md` | 🔄 In Progress | Gather ingestion metrics and disk usage benchmarks. |
| Risk analytics API design | Formalize function signatures / schemas for advanced metrics | `docs/requirements/risk_analytics_api_design.md` | ✅ Complete | Await implementation of module skeletons outlined in design. |
| Workflow regression harness | Replay reference workflows to detect JSON contract drift | (tests TBD) | ⏳ To Do | Target Phase 3 deliverable; align with CI budget. |
| Arbitrage agent rollout | Introduce cross-asset spread analysis agent | `docs/workflows/arbitrage_agent_plan.md` | ⏳ To Do | Requires new tools + prompts; track dependencies in design doc. |
| Deployment/ops guide | Document environment toggles, rollout steps, contingency plans | `docs/workflows/hard_gate_configuration_checklist.md` | ⏳ To Do | Expand into full operations manual in Phase 3. |
