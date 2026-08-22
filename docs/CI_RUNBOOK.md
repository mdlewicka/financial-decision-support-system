# Continuous integration runbook

## Purpose

The workflow in `.github/workflows/ci.yml` provides reproducible quality gates
for the Financial Decision Support System. It runs on pushes to `main`, pull
requests and manual dispatches.

The workflow does not deploy the application, publish images, call paid APIs or
require repository secrets.

## Quality gates

The workflow contains three independent jobs:

1. `Ruff quality gates` checks linting and formatting with the pinned Ruff
   version from `requirements-dev.txt`.
2. `Regression tests` installs pinned dependencies and executes non-RL and RL
   tests in separate Python processes. This preserves the local safeguard for
   native PyTorch and FAISS initialization.
3. `Container build and smoke test` validates Compose, builds both images,
   starts the local stack and checks FastAPI liveness, analytical readiness and
   Streamlit health.

Jobs use read-only repository permissions. Checkout credentials are not kept in
the working directory, concurrent obsolete runs are cancelled, and every job
has an explicit timeout.

## Local reproduction

Run the same checks from the project root before pushing:

```bash
python3 -m ruff check .
python3 -m ruff format --check .

MPLBACKEND=Agg python3 -m pytest tests -q \
  --ignore=tests/rl \
  --ignore=tests/integration/test_rl_allocation_workflow_integration.py \
  --ignore=tests/integration/test_portfolio_allocation_env_integration.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg \
python3 -m pytest \
  tests/rl \
  tests/integration/test_rl_allocation_workflow_integration.py \
  tests/integration/test_portfolio_allocation_env_integration.py -q

docker compose config --quiet
docker compose build
```

## Cost boundary

The workflow uses standard GitHub-hosted runners and local project components.
It has no paid model or data-provider integration. Public repositories can use
standard GitHub Actions runners without consuming paid minutes. Private repositories
are subject to the included Actions allowance of the repository owner's GitHub plan;
check the repository billing settings before enabling frequent private-repository
runs.

No cloud deployment is performed. Docker images remain inside the temporary CI
runner and are removed with the runner.

## Recommended repository setting

After the first successful run, optionally configure branch protection for
`main` and require these checks:

- `Ruff quality gates`;
- `Regression tests`;
- `Container build and smoke test`.

This setting is repository governance, not an application dependency.
