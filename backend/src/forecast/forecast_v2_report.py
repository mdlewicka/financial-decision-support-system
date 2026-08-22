"""Build the reproducible decision report that closes Forecasting V2."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from backend.src.config import PROJECT_ROOT

DEFAULT_STABILITY_DIRECTORY = (
    PROJECT_ROOT / "benchmark_results" / "forecast_v2" / "stability_audit"
)
DEFAULT_BENCHMARK_DIRECTORY = (
    PROJECT_ROOT / "benchmark_results" / "forecast_v2" / "model_benchmark"
)
DEFAULT_ECONOMIC_DIRECTORY = (
    PROJECT_ROOT / "benchmark_results" / "forecast_v2" / "economic_validation"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "FORECASTING_V2_RESULTS.md"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required Forecasting V2 artifact is missing: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Forecasting V2 artifact must not be empty: {path}")

    return frame


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Forecasting V2 artifact is missing: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Forecasting V2 JSON artifact must contain an object: {path}")

    return payload


def _require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Forecasting V2 artifact {path} is missing columns: {sorted(missing)}"
        )


def _one_row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column] == value]

    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one Forecasting V2 result for "
            f"{filters}, found {len(selected)}."
        )

    return selected.iloc[0]


def _finite_float(value: object, label: str) -> float:
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError(f"Forecasting V2 value must be finite: {label}")
    return result


def _integer(value: object, label: str) -> int:
    result = _finite_float(value, label)
    if not result.is_integer():
        raise ValueError(f"Forecasting V2 value must be an integer: {label}")
    return int(result)


def _yes_no(value: bool) -> str:
    return "tak" if value else "nie"


def _validate_governance(
    stability_metadata: dict[str, Any],
    benchmark_metadata: dict[str, Any],
    economic_metadata: dict[str, Any],
    candidate_decisions: pd.DataFrame,
    economic_decision: dict[str, Any],
) -> None:
    metadata_items = (
        ("audyt V1", stability_metadata),
        ("benchmark modeli", benchmark_metadata),
        ("walidacja ekonomiczna", economic_metadata),
    )

    for label, metadata in metadata_items:
        if metadata.get("test_split_read") is not False:
            raise ValueError(f"{label} must confirm that the test split was not read.")

    for label, metadata in metadata_items[1:]:
        if metadata.get("known_v1_test_used_for_selection") is not False:
            raise ValueError(
                f"{label} must confirm that the known V1 test was not used."
            )

    final_claims = candidate_decisions["decision_is_final_test_claim"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    if bool(final_claims.any()):
        raise ValueError("Development candidate decisions cannot be final-test claims.")

    if economic_decision.get("final_test_claim") is not False:
        raise ValueError("Economic screening cannot be marked as a final-test claim.")

    expected_decision = "retain_as_predictive_decision_support_only"
    if economic_decision.get("decision") != expected_decision:
        raise ValueError(
            "Economic decision must retain the classifier only as predictive "
            "decision support."
        )


def build_forecast_v2_report(
    *,
    stability_directory: Path = DEFAULT_STABILITY_DIRECTORY,
    benchmark_directory: Path = DEFAULT_BENCHMARK_DIRECTORY,
    economic_directory: Path = DEFAULT_ECONOMIC_DIRECTORY,
) -> str:
    """Return a deterministic Markdown report from committed V2 artifacts."""

    stability_summary_path = stability_directory / "stability_summary.csv"
    stability_bootstrap_path = stability_directory / "bootstrap_comparison.csv"
    stability_metadata_path = stability_directory / "audit_metadata.json"
    benchmark_summary_path = benchmark_directory / "stability_summary.csv"
    benchmark_bootstrap_path = benchmark_directory / "bootstrap_comparison.csv"
    candidate_decisions_path = benchmark_directory / "candidate_decisions.csv"
    benchmark_metadata_path = benchmark_directory / "benchmark_metadata.json"
    threshold_summary_path = economic_directory / "threshold_summary.csv"
    economic_bootstrap_path = economic_directory / "bootstrap_comparison.csv"
    economic_decision_path = economic_directory / "economic_decision.json"
    economic_metadata_path = economic_directory / "economic_validation_metadata.json"

    stability_summary = _read_csv(stability_summary_path)
    stability_bootstrap = _read_csv(stability_bootstrap_path)
    benchmark_summary = _read_csv(benchmark_summary_path)
    benchmark_bootstrap = _read_csv(benchmark_bootstrap_path)
    candidate_decisions = _read_csv(candidate_decisions_path)
    threshold_summary = _read_csv(threshold_summary_path)
    economic_bootstrap = _read_csv(economic_bootstrap_path)

    stability_metadata = _read_json(stability_metadata_path)
    benchmark_metadata = _read_json(benchmark_metadata_path)
    economic_decision = _read_json(economic_decision_path)
    economic_metadata = _read_json(economic_metadata_path)

    _require_columns(
        stability_summary,
        {
            "model",
            "folds",
            "mean_mae",
            "median_mae",
            "mean_mae_improvement_vs_zero_percent",
        },
        stability_summary_path,
    )
    _require_columns(
        stability_bootstrap,
        {"model", "bootstrap_probability_better_than_zero"},
        stability_bootstrap_path,
    )
    _require_columns(
        benchmark_summary,
        {
            "task",
            "horizon",
            "model",
            "primary_median",
            "median_improvement_vs_baseline",
            "fold_wins_vs_baseline",
            "mean_coverage",
            "mean_multiclass_brier",
            "mean_net_utility",
        },
        benchmark_summary_path,
    )
    _require_columns(
        benchmark_bootstrap,
        {
            "task",
            "horizon",
            "model",
            "comparison_model",
            "mean_primary_difference",
            "ci_lower_95",
            "ci_upper_95",
            "probability_better_than_baseline",
        },
        benchmark_bootstrap_path,
    )
    _require_columns(
        candidate_decisions,
        {"task", "horizon", "model", "decision_is_final_test_claim"},
        candidate_decisions_path,
    )
    _require_columns(
        threshold_summary,
        {
            "threshold",
            "groups",
            "mean_coverage",
            "mean_policy_return",
            "mean_benchmark_return",
            "mean_excess_return",
            "group_wins_vs_benchmark",
            "mean_drawdown_reduction",
        },
        threshold_summary_path,
    )
    _require_columns(
        economic_bootstrap,
        {
            "threshold",
            "mean_excess_return",
            "ci_lower_95",
            "ci_upper_95",
            "probability_better_than_buy_hold",
        },
        economic_bootstrap_path,
    )

    _validate_governance(
        stability_metadata,
        benchmark_metadata,
        economic_metadata,
        candidate_decisions,
        economic_decision,
    )

    audit_zero = _one_row(stability_summary, model="zero_return")
    audit_rf = _one_row(stability_summary, model="random_forest_v1")
    audit_rf_bootstrap = _one_row(stability_bootstrap, model="random_forest_v1")

    ridge_1 = _one_row(
        benchmark_summary,
        task="regression",
        horizon=1,
        model="ridge_v2",
    )
    ridge_5 = _one_row(
        benchmark_summary,
        task="regression",
        horizon=5,
        model="ridge_v2",
    )
    logistic = _one_row(
        benchmark_summary,
        task="classification",
        horizon=5,
        model="logistic_v2",
    )
    prior = _one_row(
        benchmark_summary,
        task="classification",
        horizon=5,
        model="historical_prior",
    )

    ridge_1_bootstrap = _one_row(
        benchmark_bootstrap,
        task="regression",
        horizon=1,
        model="ridge_v2",
    )
    ridge_5_bootstrap = _one_row(
        benchmark_bootstrap,
        task="regression",
        horizon=5,
        model="ridge_v2",
    )
    logistic_bootstrap = _one_row(
        benchmark_bootstrap,
        task="classification",
        horizon=5,
        model="logistic_v2",
    )

    default_threshold = _finite_float(
        economic_decision["default_threshold"],
        "default_threshold",
    )
    default_economic = _one_row(threshold_summary, threshold=default_threshold)
    default_economic_bootstrap = _one_row(
        economic_bootstrap,
        threshold=default_threshold,
    )

    folds = _integer(logistic["folds"], "classification folds")
    if folds < 5:
        raise ValueError("Forecasting V2 report requires at least five folds.")

    thresholds_evaluated = _integer(
        economic_decision["thresholds_evaluated"],
        "thresholds_evaluated",
    )
    positive_thresholds = _integer(
        economic_decision["thresholds_with_positive_mean_excess"],
        "thresholds_with_positive_mean_excess",
    )

    report = f"""# Financial Decision Support System — wyniki Forecasting V2

## Status i decyzja wykonawcza

Etap 1 został ukończony eksperymentalnie na danych rozwojowych. Powstał
powtarzalny, chronologiczny benchmark bez dostępu do zbioru testowego V1 oraz
walidacja ekonomiczna uwzględniająca koszt transakcyjny.

**Decyzja:** klasyfikator logistyczny dla horyzontu pięciu sesji zachowujemy
wyłącznie jako eksperymentalny sygnał informacyjny wspomagający interpretację.
Nie zastępuje on modeli V1, nie wyznacza wag portfela, nie zmienia mechanizmu
alokacji i nie jest dowodem rentowności strategii.

## Odpowiedź na pytanie o poprawę predykcji

Tak — uzyskano powtarzalną poprawę zdolności rozpoznawania klas kierunku w
horyzoncie pięciu sesji. Mediana balanced accuracy wzrosła z 0,333333 dla
baseline'u opartego na częstości historycznej do
{_finite_float(logistic["primary_median"], "logistic median"):.6f}. Mediana
poprawy wyniosła
{_finite_float(logistic["median_improvement_vs_baseline"], "logistic improvement"):.6f}
punktu, a model wygrał w
{_integer(logistic["fold_wins_vs_baseline"], "logistic fold wins")}/{folds}
chronologicznych foldach. Bootstrapowa różnica balanced accuracy wyniosła
{_finite_float(logistic_bootstrap["mean_primary_difference"], "logistic difference"):.6f}
z 95-procentowym przedziałem
[{_finite_float(logistic_bootstrap["ci_lower_95"], "logistic CI lower"):.6f},
{_finite_float(logistic_bootstrap["ci_upper_95"], "logistic CI upper"):.6f}], a
prawdopodobieństwo przewagi nad baseline'em wyniosło
{_finite_float(logistic_bootstrap["probability_better_than_baseline"], "logistic probability"):.3f}.

Nie uzyskano natomiast stabilnej poprawy regresji zwrotu ani przewagi
ekonomicznej po kosztach. Lepsza predykcja klas nie jest więc utożsamiana z
lepszą decyzją inwestycyjną.

## Metodyka i ochrona przed leakage

- wykorzystano wyłącznie splity `training` i `validation`;
- wykonano {folds} chronologicznych foldów expanding-window;
- cechy i modele dopasowywano wyłącznie na części treningowej każdego foldu;
- target nie przekraczał granicy części treningowej;
- test V1 odczytany: **{_yes_no(benchmark_metadata["test_split_read"])}**;
- znany wynik testu V1 wykorzystany do selekcji:
  **{_yes_no(benchmark_metadata["known_v1_test_used_for_selection"])}**;
- analiza ekonomiczna używa niepokrywających się okresów pięciosesyjnych;
- koszt zmiany pozycji wynosił
  {_finite_float(economic_metadata["config"]["transaction_cost"], "transaction cost"):.4f};
- wszystkie decyzje mają `final_test_claim=false`.

## Audyt stabilności modeli V1

| Model | Średnie MAE | Mediana MAE | Średnia poprawa MAE wobec zero-return | Prawdopodobieństwo przewagi w bootstrapie |
|---|---:|---:|---:|---:|
| Zero-return | {_finite_float(audit_zero["mean_mae"], "zero mean MAE"):.8f} | {_finite_float(audit_zero["median_mae"], "zero median MAE"):.8f} | 0,000% | — |
| Random Forest V1 | {_finite_float(audit_rf["mean_mae"], "RF mean MAE"):.8f} | {_finite_float(audit_rf["median_mae"], "RF median MAE"):.8f} | {_finite_float(audit_rf["mean_mae_improvement_vs_zero_percent"], "RF improvement"):.3f}% | {_finite_float(audit_rf_bootstrap["bootstrap_probability_better_than_zero"], "RF probability"):.3f} |

Audyt nie potwierdził stabilnej przewagi Random Forest V1 nad naiwną prognozą
zerowego zwrotu. Uzasadniało to zmianę horyzontu i zadania, ale nie usunięcie
prostych baseline'ów.

## Wyniki kandydatów V2

| Kandydat | Główna metryka | Mediana | Poprawa mediany | Wygrane foldy | Bootstrap P(lepszy) | Decyzja |
|---|---|---:|---:|---:|---:|---|
| Ridge, 1 sesja | MAE | {_finite_float(ridge_1["primary_median"], "ridge 1 median"):.8f} | {_finite_float(ridge_1["median_improvement_vs_baseline"], "ridge 1 improvement"):.3f}% | {_integer(ridge_1["fold_wins_vs_baseline"], "ridge 1 wins")}/{folds} | {_finite_float(ridge_1_bootstrap["probability_better_than_baseline"], "ridge 1 probability"):.3f} | Odrzucony |
| Ridge, 5 sesji | MAE | {_finite_float(ridge_5["primary_median"], "ridge 5 median"):.8f} | {_finite_float(ridge_5["median_improvement_vs_baseline"], "ridge 5 improvement"):.3f}% | {_integer(ridge_5["fold_wins_vs_baseline"], "ridge 5 wins")}/{folds} | {_finite_float(ridge_5_bootstrap["probability_better_than_baseline"], "ridge 5 probability"):.3f} | Nie promować |
| Logistic, 5 sesji | Balanced accuracy | {_finite_float(logistic["primary_median"], "logistic median"):.6f} | +{_finite_float(logistic["median_improvement_vs_baseline"], "logistic improvement"):.6f} pkt | {_integer(logistic["fold_wins_vs_baseline"], "logistic wins")}/{folds} | {_finite_float(logistic_bootstrap["probability_better_than_baseline"], "logistic probability"):.3f} | Tylko sygnał informacyjny |

Klasyfikator miał średnie coverage
{_finite_float(logistic["mean_coverage"], "logistic coverage"):.2%}, lecz jego
średnia screening utility była ujemna
({_finite_float(logistic["mean_net_utility"], "logistic utility"):.6f}). Ponadto
średni multiclass Brier score klasyfikatora
({_finite_float(logistic["mean_multiclass_brier"], "logistic Brier"):.6f}) był
gorszy od baseline'u prior
({_finite_float(prior["mean_multiclass_brier"], "prior Brier"):.6f}), dlatego
prawdopodobieństw nie należy przedstawiać jako skalibrowanego confidence.

## Walidacja ekonomiczna

Sprawdzono prostą politykę long/cash: pozycja długa tylko wtedy, gdy klasa `up`
ma najwyższe prawdopodobieństwo i przekracza stały próg; w przeciwnym razie
portfel pozostaje w gotówce. Nie stosowano krótkiej sprzedaży.

Dla domyślnego progu {default_threshold:.2f}:

- średnie coverage: {_finite_float(default_economic["mean_coverage"], "economic coverage"):.2%};
- średni zwrot polityki: {_finite_float(default_economic["mean_policy_return"], "policy return"):.6f};
- średni zwrot buy-and-hold: {_finite_float(default_economic["mean_benchmark_return"], "benchmark return"):.6f};
- średnia nadwyżka: {_finite_float(default_economic["mean_excess_return"], "excess return"):.6f};
- wygrane grupy fold-ticker: {_integer(default_economic["group_wins_vs_benchmark"], "group wins")}/{_integer(default_economic["groups"], "groups")};
- bootstrap P(lepszy od buy-and-hold):
  {_finite_float(default_economic_bootstrap["probability_better_than_buy_hold"], "economic probability"):.3f};
- 95-procentowy przedział nadwyżki:
  [{_finite_float(default_economic_bootstrap["ci_lower_95"], "economic CI lower"):.6f},
  {_finite_float(default_economic_bootstrap["ci_upper_95"], "economic CI upper"):.6f}];
- średnia redukcja maximum drawdown:
  {_finite_float(default_economic["mean_drawdown_reduction"], "drawdown reduction"):.2%}.

Żaden z {thresholds_evaluated} sprawdzonych progów nie osiągnął dodatniej
średniej nadwyżki względem buy-and-hold (liczba dodatnich progów:
{positive_thresholds}). Redukcja obsunięcia wynikała głównie z niskiej
ekspozycji na rynek i nie rekompensowała utraconego zwrotu.

## Ostateczne decyzje

1. Random Forest V1 pozostaje historycznym benchmarkiem, ale nie ma podstaw do
   deklarowania jego stabilnej przewagi nad zero-return.
2. Ridge V2 dla jednego dnia zostaje odrzucony.
3. Ridge V2 dla pięciu sesji nie przechodzi bramki niepewności i nie jest
   integrowany.
4. Logistic V2 dla pięciu sesji wykazuje powtarzalny sygnał klasyfikacyjny,
   dlatego może później zasilać opis kontekstu kierunkowego z możliwością
   `abstain`.
5. Logistic V2 nie może wybierać wag, zastępować Decision Engine, zmieniać
   optymalizacji portfela ani być prezentowany jako rentowna strategia.
6. Nie wykonano jednorazowego testu na nowych, rzeczywiście niewidzianych
   danych; raport nie zawiera finalnego twierdzenia o przewadze poza okresem
   rozwojowym.

## Ograniczenia

- badanie obejmuje osiem instrumentów i dane dzienne;
- klasyfikacja rozwojowa wykorzystuje pięciosesyjne cele, a osobna walidacja
  ekonomiczna dopiero usuwa nakładanie okresów;
- rozkład klas, reżimy i relacje między aktywami mogą zmieniać się w czasie;
- nie potwierdzono kalibracji prawdopodobieństw;
- strategia long/cash jest celowo prosta i nie stanowi pełnego backtestu
  portfelowego;
- nie uwzględniono wszystkich realnych ograniczeń wykonania zleceń, podatków i
  poślizgu;
- końcowy test V1 był wcześniej znany, dlatego nie mógł być testem V2;
- wiarygodne twierdzenie o generalizacji wymaga późniejszego, zamrożonego testu
  na nowych danych.

## Reprodukcja

Raport powstaje wyłącznie z wersjonowanych artefaktów:

- `benchmark_results/forecast_v2/stability_audit/`;
- `benchmark_results/forecast_v2/model_benchmark/`;
- `benchmark_results/forecast_v2/economic_validation/`.

Polecenie:

```bash
python3 -m backend.src.forecast.forecast_v2_report
```

Wynik: `docs/FORECASTING_V2_RESULTS.md`.
"""

    return report


def write_forecast_v2_report(
    *,
    stability_directory: Path = DEFAULT_STABILITY_DIRECTORY,
    benchmark_directory: Path = DEFAULT_BENCHMARK_DIRECTORY,
    economic_directory: Path = DEFAULT_ECONOMIC_DIRECTORY,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Validate committed artifacts and write the deterministic report."""

    report = build_forecast_v2_report(
        stability_directory=stability_directory,
        benchmark_directory=benchmark_directory,
        economic_directory=economic_directory,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Forecasting V2 decision report.",
    )
    parser.add_argument(
        "--stability-directory",
        type=Path,
        default=DEFAULT_STABILITY_DIRECTORY,
    )
    parser.add_argument(
        "--benchmark-directory",
        type=Path,
        default=DEFAULT_BENCHMARK_DIRECTORY,
    )
    parser.add_argument(
        "--economic-directory",
        type=Path,
        default=DEFAULT_ECONOMIC_DIRECTORY,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    path = write_forecast_v2_report(
        stability_directory=args.stability_directory,
        benchmark_directory=args.benchmark_directory,
        economic_directory=args.economic_directory,
        output_path=args.output_path,
    )
    print("Forecasting V2 decision report")
    print("Status: completed experimentally")
    print("Promoted allocation signal: no")
    print("Final-test claim: no")
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
