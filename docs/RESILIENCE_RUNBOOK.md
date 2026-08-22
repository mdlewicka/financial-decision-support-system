# Runtime status and resilience runbook

## Scope

This runbook closes the mandatory Engineering and MLOps resilience scope for
the Financial Decision Support System. All checks run locally and require no
paid API, monitoring platform or hosted language model.

## Runtime status

`GET /api/v1/system/runtime-status` returns three bounded sections:

- readiness of named analytical artifacts without filesystem paths;
- availability of optional local Ollama and deterministic fallback status;
- process-local aggregate request counts, error counts and latency.

The endpoint never returns request bodies, query strings, credentials, the
Ollama host or artifact paths. Metrics reset when the backend process restarts.
Ollama is optional and its unavailability does not fail liveness or readiness.

## Resilience evidence matrix

| Risk | Behavioural evidence |
|---|---|
| Corrupted data | `test_corrupted_cache_is_ignored`, `test_repository_rejects_corrupted_artifact` |
| NaN and infinity | decision, risk, portfolio, RL and market-service validation tests |
| Extreme parameters | decision-profile, risk-config and RL environment boundary tests |
| Prompt injection | `test_prompt_injection_cannot_replace_verified_recommendation` |
| Question outside corpus | `test_retriever_returns_empty_list_when_corpus_has_no_support` |
| Ollama unavailable | agent fallback and optional Ollama status tests |

The deterministic Decision Engine remains authoritative. LLM output cannot
replace its ranking, constraints, recommendation or portfolio weights.

## Local verification

Run the focused checks:

```bash
python3 -m pytest \
  tests/observability \
  tests/agents/test_artifact_repository.py \
  tests/agents/test_decision_narrative_agent.py \
  tests/agents/test_local_evidence_retriever.py \
  tests/api/test_agent_service_fallback.py \
  tests/market_intelligence/test_service.py \
  tests/decision/test_models.py \
  tests/risk \
  tests/portfolio \
  tests/rl/test_portfolio_allocation_env.py -q
```

Run the repository-wide quality gates using the split commands documented in
`docs/CI_RUNBOOK.md`.

## Deliberate scope boundary

The project does not add MLflow, SQLite or PostgreSQL in this stage. Frozen
JSON/CSV artifacts, metadata, tests and Git commits already provide the needed
single-user reproducibility. There is no confirmed requirement for experiment
tracking infrastructure or persistent multi-user scenario history. This keeps
the portfolio project operationally credible without introducing unused
services.
