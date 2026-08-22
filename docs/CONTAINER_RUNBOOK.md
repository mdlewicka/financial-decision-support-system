# Local container runbook

## Scope

The container setup runs the Financial Decision Support System locally as two
services:

- `backend`: FastAPI on `http://127.0.0.1:8000`;
- `frontend`: Streamlit on `http://127.0.0.1:8501`.

Both ports bind only to the local machine. The setup does not deploy the
application publicly and does not use paid APIs.

## Prerequisite

Install and start Docker Desktop. The first build downloads the pinned Python
image and Python dependencies, including the local ML stack, so it can take
several minutes and use substantial disk space.

## Build and start

Run from the project root:

```bash
docker compose build
docker compose up --detach
docker compose ps
```

Open `http://127.0.0.1:8501` in a browser. FastAPI documentation is available
at `http://127.0.0.1:8000/docs`.

The frontend starts only after the backend readiness endpoint confirms that the
core local forecast, portfolio, RAG and agent evidence is available. Docker also
checks Streamlit's built-in health endpoint.

## Health, readiness and request tracing

The system deliberately separates process liveness from analytical readiness:

- `GET /health` and `GET /api/v1/system/liveness` confirm that FastAPI is alive;
- `GET /api/v1/system/readiness` confirms that required local evidence is
  readable and non-empty;
- a readiness failure returns HTTP `503` and generic component names without
  exposing host or container filesystem paths.

Every HTTP response includes an `X-Request-ID` header. A caller can send a safe
identifier using the same header or let the backend generate one. Backend
request logs are JSON records containing only the identifier, method, URL path,
status and duration. Query parameters, request bodies and authorization headers
are not logged by this middleware.

Quick diagnostics:

```bash
curl --fail http://127.0.0.1:8000/api/v1/system/liveness
curl --fail http://127.0.0.1:8000/api/v1/system/readiness
curl --include --header 'X-Request-ID: local-check-1' \
  http://127.0.0.1:8000/api/v1/system/health
docker compose logs --tail 20 backend
```

## Logs and shutdown

```bash
docker compose logs --follow backend frontend
docker compose down
```

Use the following only when the local model and market-data caches should also
be removed:

```bash
docker compose down --volumes
```

## Optional local Ollama

Ollama is not installed inside the application images. By default, the backend
tries the host address `http://host.docker.internal:11434`. Start Ollama and the
configured model on the host if interactive generation is required.

When Ollama is unavailable, deterministic agent fallbacks remain available.
Direct generative RAG requests can report that the local language-model service
is unavailable. No request is sent to a paid cloud model.

To use another reachable Ollama address:

```bash
OLLAMA_HOST=http://host.docker.internal:11434 docker compose up --detach
```

## Reproducibility and boundaries

- The two images use the same pinned Python dependency layer.
- Containers run as the unprivileged user `app` with UID `10001`.
- Frozen benchmark artifacts and the controlled RAG corpus are copied into the
  images.
- Market and model caches use named local Docker volumes.
- Training, final-test evaluation and reinforcement-learning experiments are
  not started by Docker Compose.
- Apple MPS acceleration is not exposed inside the Linux containers; the
  application services are intended to run on CPU.
