# Financial Decision Support System

Financial Decision Support System is an AI/ML-based application for reproducible financial analysis and decision support.

The system integrates financial forecasting, explainable machine learning, risk analysis, portfolio optimization, retrieval-augmented generation, specialized AI agents and an experimental reinforcement-learning allocation component within a single decision-support architecture.

The application is designed to support interpretation and comparison of financial scenarios. It does not provide personalized investment advice and does not treat model outputs as deterministic recommendations.


Project documentation:

- [Architecture](docs/ARCHITECTURE.md)
- [Security and costs](docs/SECURITY_AND_COSTS.md)
- [Container runbook](docs/CONTAINER_RUNBOOK.md)
- [CI runbook](docs/CI_RUNBOOK.md)
- [Resilience runbook](docs/RESILIENCE_RUNBOOK.md)
- [Forecasting V2 experiment protocol](docs/FORECASTING_V2_EXPERIMENT_PROTOCOL.md)
- [Forecasting V2 results](docs/FORECASTING_V2_RESULTS.md)
- [V1 baseline](docs/V1_BASELINE.md)

## Quick start with Docker

Docker is the recommended way to run the project on another computer because it recreates the application runtime without relying on the author's local virtual environment.

Requirements:

- Docker Desktop or a compatible Docker Engine;
- Docker Compose v2 (`docker compose`).

From the project root, build and start both services:

```bash
docker compose build
docker compose up --detach
docker compose ps
```

Open the application:

- Streamlit frontend: `http://127.0.0.1:8501`
- FastAPI backend: `http://127.0.0.1:8000`

Optional health checks:

```bash
curl -fsS http://127.0.0.1:8000/api/v1/system/readiness
curl -fsS http://127.0.0.1:8501/_stcore/health
```

A healthy backend readiness response contains `"status":"ready"` and the Streamlit health endpoint returns `ok`.

Stop and remove the local containers and Compose network with:

```bash
docker compose down
```

The core analytical workflow does not require Ollama. When optional local-LLM functionality is unavailable, the application uses the implemented degraded/fallback behavior.

For additional container details, see [`docs/CONTAINER_RUNBOOK.md`](docs/CONTAINER_RUNBOOK.md).

## Project objective

The objective of the project is to build a reproducible and interpretable financial decision-support system that combines quantitative analytics with AI-assisted explanation.

The system is designed around several principles:

- reproducible market-data processing;
- chronological separation of training, validation and test data;
- explicit protection against temporal leakage;
- comparison with transparent forecasting baselines;
- interpretable model outputs;
- risk-aware portfolio analysis;
- evidence-grounded RAG responses;
- bounded responsibilities for AI agents;
- controlled experimental use of reinforcement learning;
- clear presentation of model limitations and provenance;
- separation between analytical computation and presentation logic.

The final test dataset was kept closed throughout development and model selection. It was opened once only after analytical and application decisions had been frozen.

The final evaluation is documented in:

- `benchmark_results/final_test/final_evaluation_summary_en.md`;
- `benchmark_results/final_test/final_evaluation_summary_pl.md`.

## System architecture

The web application follows a layered architecture:

```text
Streamlit frontend
        │
        ▼
HTTP / JSON
        │
        ▼
FastAPI application
        │
        ▼
Application, analytical and evidence services
        │
        ├── Forecasting
        ├── Explainability
        ├── Risk analytics
        ├── Portfolio optimization
        ├── Deterministic Decision Engine
        ├── Retrieval-Augmented Generation
        ├── Guarded local AI agents
        ├── Market Intelligence
        ├── Auditable reporting
        ├── Observability and runtime status
        └── Experimental Reinforcement Learning
```

The frontend does not reimplement financial or machine-learning calculations. It consumes structured outputs exposed by the backend API.

This separation keeps the analytical layer reproducible and allows the UI to remain focused on presentation, interpretation and interaction.

The full component, deployment and decision-flow diagrams are available in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Web application

The Streamlit interface is organized into nine main analytical modules.

### Dashboard

The Dashboard provides a compact overview of the current analytical state of the system.

It presents:

- forecasting quality indicators;
- portfolio risk and optimization metrics;
- portfolio snapshots;
- analytical component availability;
- model-quality warnings;
- evaluation-split information;
- alerts related to weak predictive performance or material portfolio risk.

The Dashboard uses existing API results and does not perform additional financial calculations.

### Forecasting

The Forecasting module presents next-session return prediction results.

The current benchmark includes:

- naive zero-return baseline;
- ARIMA;
- Random Forest.

Evaluation includes:

- MAE;
- RMSE;
- R²;
- directional accuracy;
- overall validation results;
- per-ticker results;
- observation-level forecast information.

The forecasting target is the next-session simple return.

Model development and comparison are performed using the validation dataset.

After development decisions were frozen, the selected forecasting configurations were evaluated once on the final test split. The test results are treated as terminal evaluation results and are not used for further model selection or tuning.

### Risk and Scenarios

The Risk and Scenarios module presents market and portfolio risk information.

The implemented analytical layer includes:

- annualized return;
- annualized volatility;
- Value at Risk;
- Conditional Value at Risk;
- maximum drawdown;
- correlation analysis;
- concentration measures;
- effective number of assets;
- diversification measures;
- predefined portfolio scenarios.

The interface supports comparison of alternative portfolio profiles and examination of detailed risk characteristics.

### Optimization

The Optimization module presents portfolio construction and allocation results.

Available portfolio strategies include:

- defensive allocation;
- equal-weight allocation;
- growth allocation;
- maximum-Sharpe allocation;
- minimum-volatility allocation.

The interface presents:

- validation metrics;
- portfolio weights;
- asset-level return contributions;
- asset-level volatility contributions;
- cumulative portfolio returns;
- training-versus-validation comparison.

Portfolio visualizations use Plotly for interactive analytical charts.

Optimized strategies are constructed using training data and evaluated on validation data.

For the final test evaluation, previously frozen portfolio weights were reused without re-optimization.

### Decision Workspace

The human-in-the-loop Decision Workspace converts a typed user profile and
explicit assumptions into a deterministic ranking of frozen portfolio
scenarios. It presents hard-constraint violations, normalized component scores,
trade-offs, provenance and a before-versus-after comparison when assumptions
change.

The optional local LLM explains an already computed result. It cannot select
weights, modify analytical facts or replace the user's final decision. When
Ollama is unavailable, a deterministic fallback preserves the workflow.

### Market Intelligence

Market Intelligence provides a small current-context snapshot with the latest
available price, daily change, UTC timestamp, source, market timezone and
freshness/cache state. It supports manual refresh and explicit degraded/offline
behavior. It is not a streaming feed or an order-execution service.

### AI Insights

The AI Insights module combines Retrieval-Augmented Generation and specialized analytical agents.

The RAG component provides evidence-grounded answers using a controlled financial knowledge base.

The agent layer exposes specialized workflows for:

- forecasting interpretation;
- risk interpretation;
- portfolio allocation interpretation.

Agent responses contain structured analytical sections such as:

- summary;
- findings;
- risks;
- recommendations;
- limitations;
- evidence.

Generative requests use a longer frontend timeout than normal analytical API requests because local LLM inference may require additional processing time.

### Explainability and Governance

The Explainability and Governance module provides model interpretation and methodological controls.

The current XAI artifact focuses on the Random Forest forecasting model and includes:

- global SHAP feature importance;
- local SHAP explanations;
- Partial Dependence analysis;
- Accumulated Local Effects;
- representative local observations;
- model and artifact metadata;
- explicit interpretation limitations.

Interactive Plotly charts are used for SHAP, PDP and ALE visualizations.

The module explicitly distinguishes predictive explanation from causal inference.

SHAP, PDP and ALE explain model behavior and must not be interpreted as proof of causal market relationships.

### Reports

The Reports module generates reproducible analytical summaries from existing API outputs.

Reports currently include information from:

- system metadata;
- forecasting;
- explainability;
- risk analysis;
- portfolio optimization;
- RAG;
- AI agents.

Generated reports can be downloaded as:

- Markdown;
- JSON.

Report generation does not retrain models or recalculate analytical results. It collects already available structured outputs from the backend.

## Market-data pipeline

The market-data workflow currently supports eight instruments:

| Ticker | Representation |
|---|---|
| `EPOL` | Polish equities |
| `VGK` | European equities |
| `SPY` | large-cap US equities |
| `VT` | global equities |
| `QQQ` | Nasdaq-100 equities |
| `GLD` | gold |
| `BTC-USD` | Bitcoin |
| `AGG` | US aggregate bonds |

The pipeline includes:

- historical data acquisition through Yahoo Finance;
- schema standardization;
- data validation;
- chronological sorting;
- feature engineering;
- minimum-observation validation;
- explicit target dates;
- next-session return construction;
- chronological training, validation and test splits;
- temporal-leakage protection;
- reproducibility metadata.

The dataset workflow is designed so that target dates cannot cross split boundaries.

## Forecasting methodology

The forecasting problem is defined as prediction of the next-session simple return.

The implemented comparison contains three model families:

```text
Naive baseline
ARIMA
Random Forest
```

The naive model predicts zero return and therefore acts as a transparent reference point.

Random Forest is also used by the current XAI layer.

The evaluation process uses validation data for model development and comparison.

The final test split was intentionally excluded from model selection and interface development. After the forecasting configurations were frozen, it was used once for terminal out-of-sample evaluation.

The final test results must not be used for additional model selection, hyperparameter tuning or ARIMA order selection.

### Final forecasting results

The frozen forecasting approaches were evaluated once on the final test split.

| Model | Observations | MAE | RMSE | R² | Directional accuracy |
|---|---:|---:|---:|---:|---:|
| Random Forest | 4786 | 0.009504 | 0.014632 | 0.007772 | 54.53% |
| ARIMA walk-forward | 4786 | 0.009518 | 0.014706 | -0.002362 | 53.36% |
| Naive zero-return baseline | 4786 | 0.009534 | 0.014707 | -0.002547 | 0.42% |

Random Forest achieved the strongest overall result among the evaluated forecasting models.

Its advantage remains limited. The test R² is still close to zero, which supports a conservative interpretation of short-horizon financial forecasting performance.

Forecast outputs should therefore be treated as supporting analytical signals rather than stand-alone investment rules.

## Explainable AI

The active explainability workflow uses:

```text
SHAP
Partial Dependence
Accumulated Local Effects
```

Global explanations identify features most strongly associated with model output.

Local explanations show how individual features contribute to one representative prediction.

Partial Dependence and ALE are used to inspect model response patterns across feature values.

Important methodological restrictions apply:

- SHAP values describe model behavior rather than causality;
- PDP can be distorted when predictors are strongly correlated;
- ALE reduces some correlation-related problems but remains a model-interpretation technique;
- strong explainability does not compensate for weak predictive performance;
- validation explanations do not guarantee future model stability.

The XAI workflow is based on the previously selected Random Forest configuration and does not use the final test split for model-selection purposes.

## Portfolio and risk analytics

Portfolio evaluation contains both rule-based and optimized strategies.

The analytical framework includes:

```text
expected returns
covariance estimation
Markowitz-style optimization
portfolio weights
annualized risk and return
Sharpe ratio
Sortino ratio
maximum drawdown
VaR
CVaR
concentration
effective number of assets
diversification ratio
asset-level contributions
cumulative portfolio returns
```

Optimized portfolio parameters are derived from training data.

Primary application results are reported for the validation dataset.

This separation makes it possible to examine whether a strategy remains stable outside the period used for its construction.

For the one-time final evaluation, the portfolio weights previously constructed before test access were frozen and applied directly to the test period without re-optimization.

### Final classical portfolio results

| Strategy | Annualized return | Annualized volatility | Sharpe | Sortino | VaR | CVaR | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Defensive | 16.63% | 10.56% | 1.388 | 2.044 | 1.04% | 1.54% | 8.01% |
| Equal Weight | 19.78% | 15.34% | 1.160 | 1.744 | 1.39% | 2.10% | 11.84% |
| Growth | 19.67% | 15.87% | 1.115 | 1.676 | 1.42% | 2.18% | 14.51% |
| Maximum Sharpe | 15.98% | 14.27% | 0.981 | 1.464 | 1.41% | 1.93% | 11.38% |
| Minimum Volatility | 17.13% | 10.65% | 1.422 | 2.037 | 1.06% | 1.62% | 9.11% |

The final test results show that:

- `equal_weight` achieved the highest annualized return among classical strategies;
- `minimum_volatility` achieved the highest Sharpe ratio;
- `defensive` achieved the lowest realized volatility;
- `defensive` also achieved the smallest maximum drawdown;
- `maximum_sharpe` did not retain the highest Sharpe ratio on unseen test data.

These results illustrate the limitations of historical portfolio optimization and the sensitivity of strategy rankings to changing market conditions.

## Retrieval-Augmented Generation

The compact RAG implementation supports:

- deterministic document processing;
- Sentence Transformer embeddings;
- FAISS vector retrieval;
- BM25 lexical retrieval;
- hybrid retrieval;
- reranking;
- local answer generation through Ollama;
- cited evidence;
- benchmark evaluation;
- retrieval metrics;
- machine-readable benchmark artifacts.

The current development corpus contains a small controlled set of financial reference documents and a manually defined golden evaluation dataset.

The current benchmark is intended primarily for system development and configuration comparison.

The evaluated retrieval strategies are:

```text
Vector
Hybrid
Reranked
```

On the current benchmark, all three configurations achieved the same aggregate retrieval quality score. The more complex retrieval configurations did not demonstrate a measurable quality advantage on the small controlled corpus.

The current RAG benchmark remains a development-oriented evaluation based on the controlled golden dataset. It was not part of the one-time market test-split evaluation used for forecasting, portfolio and RL assessment.

A broader RAG evaluation would require an expanded, independently prepared evaluation corpus.

The generation layer is instructed to answer in the language of the user question while grounding the answer in retrieved evidence.

## AI agents

The system contains a controlled agent layer built around specialized analytical responsibilities.

The current agent workflows cover:

```text
Forecast Agent
Risk Agent
Allocation Agent
```

Each agent receives a bounded analytical task and returns a structured result.

The common response contract contains:

```text
agent_name
question
summary
findings
risks
recommendations
limitations
evidence
```

The design intentionally avoids an unrestricted autonomous agent swarm.

Agents are expected to interpret existing analytical evidence rather than replace deterministic financial calculations.

The recorded agent artifacts were produced using validation-based analytical evidence and explicitly indicate that the final test split was not used for agent development.

## Reinforcement-learning extension

The repository contains a reinforcement-learning allocation research extension based on Proximal Policy Optimization.

The RL environment supports:

- dynamic portfolio allocation;
- transaction costs;
- turnover tracking;
- maximum asset-weight constraints;
- drawdown penalties;
- portfolio-value tracking;
- reproducible evaluation trajectories.

The PPO policy is trained using training data and was initially evaluated on validation data.

The RL extension is treated as an experimental portfolio-allocation component rather than as a replacement for the main deterministic portfolio and risk workflows.

### Final PPO result

The PPO model was frozen before final-test access and loaded from the previously saved model artifact without retraining.

Final-test PPO metrics were:

| Metric | Result |
|---|---:|
| Observations | 565 |
| Annualized mean return | 28.93% |
| Annualized volatility | 22.25% |
| Sharpe ratio | 1.211 |
| Sortino ratio | 2.649 |
| Value at Risk | 1.38% |
| Conditional Value at Risk | 2.07% |
| Maximum drawdown | 11.63% |
| Cumulative return | 81.67% |
| Final portfolio value | 1.8167 |
| Total transaction cost | 0.005061 |
| Mean turnover | 0.008958 |
| Total reward | -2.569404 |

PPO achieved the highest annualized return among the evaluated allocation approaches and also achieved the highest Sortino ratio.

However, it did not achieve the highest Sharpe ratio and did not minimize volatility or maximum drawdown.

The result therefore supports PPO as a promising experimental extension, but does not establish universal superiority over simpler and more interpretable classical portfolio methods.

The negative total reward does not imply a negative portfolio return. The RL reward is an environment-specific optimization signal that incorporates penalties and is not equivalent to investment return.

## Final evaluation

The project uses a controlled one-time final-evaluation workflow.

Before test access:

- model and strategy choices were frozen;
- final benchmark artifacts were fingerprinted using SHA256;
- a frozen manifest was created;
- the full project passed Ruff;
- the non-RL regression suite passed with 499 tests;
- the RL and RL integration suite passed with 59 tests;
- the final-evaluation test suite passed;
- a dry-run preflight reported `READY`.

The frozen manifest is stored at:

```text
benchmark_results/final_evaluation_frozen_manifest.json
```

The final market test split was then opened exactly once through the controlled executor.

The successful run produced:

```text
benchmark_results/final_test/final_test_usage.json
benchmark_results/final_test/final_evaluation_result.json
benchmark_results/final_test/forecast/
benchmark_results/final_test/portfolio/
benchmark_results/final_test/rl/
```

No `final_evaluation_failure.json` was produced.

Human-readable summaries are available in:

```text
benchmark_results/final_test/final_evaluation_summary_en.md
benchmark_results/final_test/final_evaluation_summary_pl.md
```

The final test split is now considered consumed.

Its results must not be used for:

- further model selection;
- hyperparameter tuning;
- ARIMA order selection;
- portfolio re-optimization;
- PPO retraining;
- reward-function redesign;
- selecting a strategy and then presenting the same test result as an independent confirmation.

Any future methodological changes require a new evaluation protocol and genuinely unseen data.

## Analytical notebooks

The project contains six analytical notebooks:

```text
01_data_exploration.ipynb
02_forecasting.ipynb
03_risk_and_portfolio.ipynb
04_explainable_ai.ipynb
05_llm_rag_agents.ipynb
06_reinforcement_learning.ipynb
```

The notebooks document the major analytical components of the project.

Their roles are:

- `01_data_exploration.ipynb` — exploratory analysis and dataset structure;
- `02_forecasting.ipynb` — forecasting baselines and model comparison;
- `03_risk_and_portfolio.ipynb` — portfolio risk and allocation analysis;
- `04_explainable_ai.ipynb` — SHAP, PDP and ALE interpretation;
- `05_llm_rag_agents.ipynb` — RAG benchmark and agent architecture;
- `06_reinforcement_learning.ipynb` — PPO allocation experiment and comparison with classical strategies.

The notebooks primarily analyze reproducible backend artifacts rather than reimplement the analytical logic.

## API

The FastAPI application exposes analytical capabilities through versioned endpoints.

Main endpoint groups include:

```text
/api/v1/forecast
/api/v1/xai
/api/v1/risk
/api/v1/portfolio
/api/v1/rag
/api/v1/agents
/api/v1/decision
/api/v1/reports
/api/v1/market-intelligence
/api/v1/system
```

The application title is:

```text
Financial Decision Support API
```

FastAPI acts as the boundary between the analytical backend and the Streamlit user interface.

## Technology stack

The project uses:

```text
Python 3.12
FastAPI
Streamlit
Pandas
NumPy
scikit-learn
statsmodels
SHAP
Plotly
FAISS
Sentence Transformers
BM25
Ollama
PyTorch
Gymnasium
Stable-Baselines3
Pytest
Ruff
Docker Compose
GitHub Actions
```

## Project structure

```text
project-root/
├── backend/
│   └── src/
│       ├── agents/
│       ├── api/
│       ├── decision/
│       ├── domain/
│       ├── final_evaluation/
│       ├── forecast/
│       ├── market_intelligence/
│       ├── models/
│       ├── observability/
│       ├── pipelines/
│       ├── portfolio/
│       ├── providers/
│       ├── rag/
│       ├── reporting/
│       ├── risk/
│       ├── rl/
│       ├── services/
│       ├── storage/
│       ├── xai/
│       ├── config.py
│       ├── data_cli.py
│       └── main.py
├── frontend/
│   ├── ai_insights_view.py
│   ├── api_client.py
│   ├── app.py
│   ├── branding.py
│   ├── dashboard_view.py
│   ├── decision_workspace_view.py
│   ├── forecast_view.py
│   ├── governance_view.py
│   ├── i18n.py
│   ├── market_intelligence_view.py
│   ├── optimization_view.py
│   ├── reports_view.py
│   └── risk_view.py
├── benchmark_results/
│   ├── agents/
│   ├── final_test/
│   ├── forecast/
│   ├── rag/
│   ├── risk/
│   ├── rl/
│   └── xai/
├── data/
├── docs/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_forecasting.ipynb
│   ├── 03_risk_and_portfolio.ipynb
│   ├── 04_explainable_ai.ipynb
│   ├── 05_llm_rag_agents.ipynb
│   └── 06_reinforcement_learning.ipynb
├── tests/
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── compose.yaml
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

Generated datasets and benchmark artifacts are treated as reproducible analytical outputs.

Final-test artifacts additionally act as methodological records of the terminal evaluation and should not be regenerated from the consumed test split.

## Local setup

The project requires Python 3.12.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Install development dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
```

Verify the environment:

```bash
python3 -m pip check
```

Some RAG and agent workflows additionally require a locally available Ollama installation and the configured local language model.

## Rebuild market datasets

The market-data workflow can be executed through the data CLI.

The end date is exclusive and should be specified explicitly for reproducibility.

Example:

```bash
python3 -m backend.src.data_cli \
  --start-date 2010-01-01 \
  --end-date 2026-08-01 \
  --minimum-observations 1000
```

Generated raw, processed and model datasets are local analytical artifacts.

Rebuilding datasets after the terminal final evaluation creates a new methodological state and should not be used to claim continuity with the recorded final-test result.

## Run forecasting workflows

Forecasting benchmark functionality is available through the backend forecasting package.

For example:

```bash
python3 -m backend.src.forecast.cli
```

Benchmark artifacts can contain observation-level predictions, aggregate metrics and machine-readable evaluation summaries.

The standard forecasting workflows operate on validation data for development purposes.

## Run the RAG benchmark

Example:

```bash
python3 -m backend.src.rag.cli \
  --corpus data/rag/evaluation/reference_corpus.json \
  --golden-dataset data/rag/evaluation/golden_dataset_expanded.json \
  --output benchmark_results/rag
```

RAG execution may require locally available embedding, reranking and Ollama models depending on the selected configuration.

## Run the web application without Docker

Start the FastAPI backend from the project root:

```bash
python3 -m uvicorn \
  backend.src.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

In a second terminal, activate the same virtual environment and start Streamlit:

```bash
cd /path/to/project-root
source .venv/bin/activate

PYTHONPATH="$PWD" python3 -m streamlit run frontend/app.py
```

By default, Streamlit is available locally on port `8501`.

## Run frontend tests

Run the frontend test suite with:

```bash
python3 -m pytest \
  tests/frontend \
  -q
```

## Run project quality checks

Run Ruff linting:

```bash
python3 -m ruff check .
```

Check formatting:

```bash
python3 -m ruff format --check .
```

## Regression testing

The project contains both conventional analytical tests and reinforcement-learning tests.

Because native PyTorch and FAISS initialization can interact poorly when the entire suite is loaded into a single Python process on some environments, regression testing is performed in two controlled groups.

Run the non-RL regression suite:

```bash
python3 -m pytest tests -q \
  --ignore=tests/rl \
  --ignore=tests/integration/test_rl_allocation_workflow_integration.py \
  --ignore=tests/integration/test_portfolio_allocation_env_integration.py
```

Then run the RL regression suite separately:

```bash
python3 -m pytest \
  tests/rl \
  tests/integration/test_rl_allocation_workflow_integration.py \
  tests/integration/test_portfolio_allocation_env_integration.py \
  -q
```

The current V2 closure regression state is:

```text
Non-RL: 793 passed
RL:      59 passed
Total:  852 passed
```

The non-RL suite also reported four non-blocking deprecation warnings originating from external dependencies.

This test split is an execution-environment safeguard and does not change the logical coverage of the test suite.

## Continuous integration

GitHub Actions runs Ruff linting, formatting checks, the split non-RL and RL
regression suites, and a local Docker Compose build and smoke test. The workflow
uses read-only repository permissions and does not require secrets, deploy the
application, publish images or call paid APIs.

See [`docs/CI_RUNBOOK.md`](docs/CI_RUNBOOK.md) for the exact quality gates,
local reproduction commands, security boundaries and GitHub Actions cost notes.

## Methodological principles

The project follows the following evaluation rules:

```text
Training data
    ↓
model fitting / parameter construction

Validation data
    ↓
development, comparison and interface analysis

Test data
    ↓
one final evaluation after development decisions are frozen
```

This protocol was followed during development.

The final test split remained closed until all relevant forecasting, portfolio and reinforcement-learning decisions had been frozen.

The split was then opened exactly once through the controlled final-evaluation workflow.

A persistent usage marker records that the test split has been consumed:

```text
benchmark_results/final_test/final_test_usage.json
```

The test split must not be treated as an independently reusable evaluation dataset.

Other methodological principles include:

- chronological rather than random financial-data splitting;
- explicit target dates;
- prevention of target leakage across dataset boundaries;
- transparent comparison with a naive baseline;
- reporting of negative or weak model results;
- separation between predictive performance and interpretability;
- validation of portfolio behavior outside the construction period;
- grounding of generated answers in retrievable evidence;
- explicit disclosure of analytical limitations;
- reproducible benchmark artifacts;
- separation of backend computation from frontend presentation;
- explicit separation between model selection and final evaluation;
- preservation of final-test governance records.

## Interpretation of results

The system is a decision-support application rather than an automated investment adviser.

Forecast accuracy can be weak even when explanations are technically valid.

Historical portfolio optimization does not guarantee future performance.

Risk estimates depend on historical observations and assumptions.

RAG outputs depend on the quality and coverage of the available knowledge base.

Agent outputs depend on both structured analytical evidence and local language-model behavior.

Reinforcement-learning results can depend strongly on the training regime, reward design, market regime and environment configuration.

A strong result on one terminal test period does not establish universal future superiority.

All outputs should therefore be interpreted together with their provenance, evaluation split and stated limitations.

## Current limitations

The current system has several intentional limitations.

The predictive task is difficult and the forecasting models provide only limited predictive advantage, despite Random Forest achieving the strongest result among the evaluated forecasting approaches on the final test split.

The RAG evaluation corpus is still small and remains primarily development-oriented.

The final market test dataset has already been consumed by the one-time terminal evaluation and must not be reused for further model development or tuning.

Portfolio optimization relies on historical return and covariance relationships that may change over time.

The PPO result was obtained on one final test period and should not be interpreted as evidence of universal superiority over classical methods.

The application does not currently model all investor-specific constraints, taxes or a complete real-world transaction-cost framework.

Local LLM execution introduces latency and depends on the locally available model.

The application is intended for analytical, educational and research use and does not constitute financial advice.

## Project completion status

The main engineering and analytical implementation stages are complete:

```text
1. V1 analytical baseline and terminal evaluation            COMPLETE
2. Forecasting V2 and economic promotion gates               COMPLETE
3. Deterministic Decision Engine and HITL workspace           COMPLETE
4. Grounded AI, XAI, RAG and auditable reporting              COMPLETE
5. Market context, localization and accessibility             COMPLETE
6. Containers, CI, observability and resilience               COMPLETE
7. Project documentation                                      COMPLETE
8. Final package verification and Docker smoke test            COMPLETE
```

The final test split has been consumed by the one-time terminal evaluation and is no longer available for unbiased model development.

Any future methodological changes would require a new evaluation protocol and genuinely unseen data.

## Academic and engineering focus

The project demonstrates both Data Science methodology and AI/ML Engineering practices.

Its primary engineering and methodological contributions include:

- reproducible data workflows;
- typed analytical services;
- explicit API contracts;
- automated unit and integration testing;
- leakage-aware temporal evaluation;
- forecasting baseline comparison;
- model explainability;
- quantitative risk analysis;
- portfolio optimization;
- retrieval evaluation;
- evidence-grounded generation;
- controlled agent orchestration;
- experimental reinforcement learning;
- interactive analytical visualization;
- reproducible benchmark artifacts;
- frozen-artifact fingerprinting;
- controlled one-time final evaluation;
- explicit test-split governance;
- clear separation between application layers;
- explicit governance and limitations.

The goal is not to maximize architectural complexity, but to build a coherent, testable and interpretable financial decision-support system.
