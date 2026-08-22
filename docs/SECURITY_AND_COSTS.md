# Security, privacy and operating-cost boundary

## Executive statement

The implemented default workflow is local-first and does not require a paid
language-model API, cloud account or secret model key. Core analytics, portfolio
ranking and deterministic explanations remain available without Ollama or
network access to current market data.

This document describes the current project boundary. It is not a claim of
enterprise-grade security or a promise that every optional external service will
remain free under every future usage model.

## External dependencies

| Dependency | Required for core decision flow | Credential in current implementation | Direct per-call fee in the implemented path | Degraded behavior |
|---|---:|---:|---:|---|
| Ollama with local model | no | no | no paid API call | deterministic narrative fallback |
| Yahoo Finance current snapshot | no | no | no project API billing | cached/offline status is shown |
| Hugging Face model download | only when local embedding assets are absent | no by default | no project API billing | use already cached model or report missing dependency |
| Docker Desktop / engine | no | no | no application API billing | run Python services directly |
| GitHub Actions | no for local use | repository permissions only | depends on repository/account quota | reproduce gates locally |

“No direct per-call fee” does not mean zero total cost. Local inference and
containers consume CPU/GPU, memory, disk space and electricity. Internet access,
public hosting and future provider-plan changes may introduce costs.

## Secrets and private data

The repository quality boundary requires that common secret files, private
keys, credentials and local absolute paths are not tracked. `.env` files remain
ignored. Before sharing or publishing the repository, run the project quality
and regression checks, review the tracked file set, and verify that no secrets,
private keys, credentials, `.env` contents or machine-specific absolute paths
have been added.

If a future deployment introduces authentication, a paid data provider or a
hosted LLM, credentials must be supplied by the deployment environment or a
secret manager. They must never be committed to source control, notebooks,
screenshots, reports or container images.

## LLM and agent boundary

- the LLM receives bounded context for explanation, not allocation authority;
- deterministic code computes rankings and facts;
- cited evidence and factual guardrails are applied after generation;
- operation without the LLM is tested;
- hidden chain-of-thought is not requested for storage or written to artifacts;
- prompts and retrieved text are treated as untrusted inputs;
- prompt-injection and unsupported-question behavior have regression tests.

Local inference reduces third-party data exposure, but it does not make arbitrary
input safe. A public multi-user deployment would additionally require request
size limits, authentication, authorization, rate limiting, audit retention rules
and abuse monitoring.

## Application and container controls

- FastAPI validates typed request schemas;
- invalid, missing and non-finite analytical data is rejected or surfaced;
- health and readiness endpoints have separate meanings;
- request ID, latency and error information support diagnosis;
- Docker services run as a non-root application user;
- only the backend and frontend ports are exposed by the local Compose setup;
- no automatic market order or trading credential integration exists;
- CI uses read-only repository permissions and does not deploy or publish images.

## Financial safety boundary

The application is educational and analytical. Historical estimates, model
predictions, optimized portfolios and generated narratives are uncertain. The
system does not know a user's full financial situation, taxes, legal constraints
or suitability requirements and must not be presented as personalized financial
advice.

## Public deployment decision

Public hosting is conditional, not required for the portfolio evidence. Before
deployment, make an explicit decision about:

1. authentication and rate limits;
2. allowed input and retained logs;
3. model and data-provider terms;
4. monthly compute, storage and network budget;
5. denial-of-service and prompt-abuse controls;
6. privacy notice and deletion policy;
7. whether the demo should disable generative inference entirely.

A screen recording and reproducible Docker instructions can demonstrate the
system without accepting the risk or recurring cost of an internet-facing
service.
