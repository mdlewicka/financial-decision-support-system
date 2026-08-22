import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentArtifactPaths:
    forecast: Path
    xai: Path
    portfolio: Path
    reinforcement_learning: Path

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
    ) -> "AgentArtifactPaths":
        benchmark_root = project_root / "benchmark_results"
        return cls(
            forecast=(
                benchmark_root / "forecast" / "random_forest" / "forecast_result.json"
            ),
            xai=(benchmark_root / "xai" / "random_forest" / "xai_result.json"),
            portfolio=(benchmark_root / "risk" / "portfolio" / "portfolio_result.json"),
            reinforcement_learning=(benchmark_root / "rl" / "ppo" / "rl_result.json"),
        )


class DecisionSupportArtifactRepository:
    """Load and compact reproducible benchmark artifacts for agent prompts."""

    def __init__(self, paths: AgentArtifactPaths) -> None:
        self.paths = paths
        self._forecast = self._read_json(paths.forecast)
        self._xai = self._read_json(paths.xai)
        self._portfolio = self._read_json(paths.portfolio)
        self._rl = self._read_json(paths.reinforcement_learning)
        self._validate_contracts()

    def forecast_context(
        self,
        ticker: str | None = None,
        top_features: int = 10,
    ) -> dict[str, Any]:
        if top_features <= 0:
            raise ValueError("top_features must be greater than zero.")

        metrics = self._mapping(self._forecast, "metrics")
        overall = self._mapping(metrics, "overall")
        by_ticker = self._mapping(metrics, "by_ticker")
        selected_ticker = ticker.strip().upper() if ticker else None
        ticker_metrics: dict[str, Any] | None = None

        if selected_ticker is not None:
            if selected_ticker not in by_ticker:
                raise ValueError(f"Unknown forecast ticker: {selected_ticker}.")
            ticker_metrics = self._mapping(
                by_ticker,
                selected_ticker,
            )

        global_shap = self._mapping(self._xai, "global_shap")
        contributions = self._mapping(
            global_shap,
            "mean_absolute_contributions",
        )
        ranked_features = sorted(
            (
                {
                    "feature": str(feature),
                    "mean_absolute_shap_value": float(value),
                }
                for feature, value in contributions.items()
            ),
            key=lambda row: row["mean_absolute_shap_value"],
            reverse=True,
        )[:top_features]

        return {
            "model_name": self._forecast["model_name"],
            "evaluation_split": self._forecast["evaluation_split"],
            "target_definition": self._forecast["target_definition"],
            "overall_metrics": overall,
            "selected_ticker": selected_ticker,
            "ticker_metrics": ticker_metrics,
            "xai_metadata": self._mapping(self._xai, "metadata"),
            "top_global_shap_features": ranked_features,
        }

    def risk_context(self) -> dict[str, Any]:
        return {
            "return_definition": self._portfolio["return_definition"],
            "validation_market": self._mapping(
                self._mapping(self._portfolio, "market_splits"),
                "validation",
            ),
            "validation_scenarios": self._scenario_rows(include_weights=False),
        }

    def xai_context(
        self,
        top_features: int = 5,
    ) -> dict[str, Any]:
        """Return compact global XAI evidence with an explicit scope boundary."""
        if top_features <= 0:
            raise ValueError("top_features must be greater than zero.")
        metadata = self._mapping(self._xai, "metadata")
        global_shap = self._mapping(self._xai, "global_shap")
        contributions = self._mapping(global_shap, "mean_absolute_contributions")
        ranked_features = sorted(
            (
                {
                    "feature": str(feature),
                    "mean_absolute_shap_value": float(value),
                }
                for feature, value in contributions.items()
            ),
            key=lambda row: row["mean_absolute_shap_value"],
            reverse=True,
        )[:top_features]
        partial_dependence = self._optional_mapping(
            self._xai,
            "partial_dependence",
        )
        accumulated_local_effects = self._optional_mapping(
            self._xai,
            "accumulated_local_effects",
        )
        return {
            "model_name": metadata.get("model_name"),
            "explanation_split": metadata.get("explanation_split"),
            "target_definition": metadata.get("target_definition"),
            "global_observations": global_shap.get("observations"),
            "top_global_shap_features": ranked_features,
            "partial_dependence_features": sorted(partial_dependence),
            "accumulated_local_effect_features": sorted(accumulated_local_effects),
            "scope": (
                "Forecast-model interpretation only; this evidence does not "
                "determine or modify the portfolio Decision Engine ranking."
            ),
        }

    def allocation_context(
        self,
        risk_preference: str,
    ) -> dict[str, Any]:
        normalized_preference = risk_preference.strip().lower()
        if normalized_preference not in {
            "defensive",
            "balanced",
            "growth",
        }:
            raise ValueError("risk_preference must be defensive, balanced or growth.")

        rl_comparison = self._rl.get("comparison")
        if not isinstance(rl_comparison, list) or not rl_comparison:
            raise ValueError("RL comparison must be a non-empty list.")

        return {
            "risk_preference": normalized_preference,
            "portfolio_scenarios": self._scenario_rows(include_weights=True),
            "rl_evaluation_split": self._rl["evaluation_split"],
            "rl_test_split_used": self._rl["test_split_used"],
            "rl_environment_config": self._mapping(
                self._rl,
                "environment_config",
            ),
            "strategy_comparison": rl_comparison,
        }

    def _scenario_rows(
        self,
        include_weights: bool,
    ) -> list[dict[str, Any]]:
        scenarios = self._portfolio.get("scenarios")
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("Portfolio scenarios must be a non-empty list.")

        rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                raise TypeError("Each portfolio scenario must be an object.")
            validation = self._mapping(scenario, "validation")
            row: dict[str, Any] = {
                "name": scenario["name"],
                "construction": scenario["construction"],
                "validation_metrics": self._mapping(
                    validation,
                    "metrics",
                ),
                "concentration_index": validation["concentration_index"],
                "effective_number_of_assets": validation["effective_number_of_assets"],
                "diversification_ratio": validation["diversification_ratio"],
            }
            if include_weights:
                row["weights"] = self._mapping(scenario, "weights")
            rows.append(row)
        return rows

    def _validate_contracts(self) -> None:
        self._require_keys(
            self._forecast,
            {
                "model_name",
                "evaluation_split",
                "target_definition",
                "metrics",
            },
            "forecast",
        )
        self._require_keys(
            self._xai,
            {"metadata", "global_shap", "local_shap"},
            "xai",
        )
        self._require_keys(
            self._portfolio,
            {"return_definition", "market_splits", "scenarios"},
            "portfolio",
        )
        self._require_keys(
            self._rl,
            {
                "evaluation_split",
                "test_split_used",
                "environment_config",
                "comparison",
            },
            "reinforcement_learning",
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Artifact does not exist: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Artifact is not valid JSON: {path}") from error
        if not isinstance(value, dict):
            raise TypeError(f"Artifact must contain a JSON object: {path}")
        return value

    @staticmethod
    def _mapping(
        value: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        nested = value.get(key)
        if not isinstance(nested, dict):
            raise TypeError(f"{key} must be a JSON object.")
        return nested

    @staticmethod
    def _optional_mapping(
        value: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        nested = value.get(key, {})
        if not isinstance(nested, dict):
            raise TypeError(f"{key} must be a JSON object.")
        return nested

    @staticmethod
    def _require_keys(
        value: dict[str, Any],
        required: set[str],
        label: str,
    ) -> None:
        missing = required.difference(value)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{label} artifact is missing: {names}.")
