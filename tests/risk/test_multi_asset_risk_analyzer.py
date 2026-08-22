from collections.abc import Callable

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.risk.risk_analyzer import (
    MultiAssetRiskAnalysisConfig,
    MultiAssetRiskAnalyzer,
)
from backend.src.risk.risk_metrics import MultiAssetRiskResult


def create_market_data() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
        ]
    )

    prices = {
        "SPY": [
            100.0,
            101.0,
            100.0,
            102.0,
            103.0,
        ],
        "QQQ": [
            200.0,
            204.0,
            202.0,
            206.0,
            205.0,
        ],
        "BTC-USD": [
            40000.0,
            41000.0,
            40500.0,
            42000.0,
            41800.0,
        ],
    }

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for (
        ticker,
        ticker_prices,
    ) in prices.items():
        for (
            date_value,
            price,
        ) in zip(
            dates,
            ticker_prices,
            strict=True,
        ):
            rows.append(
                {
                    "date": date_value,
                    "ticker": ticker,
                    "adjusted_close": price,
                }
            )

    return pd.DataFrame(rows)


def test_analyze_builds_complete_matrices_from_shared_dates() -> None:
    market_data = create_market_data()

    analyzer = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(
            periods_per_year=252,
            minimum_common_observations=4,
        )
    )

    result = analyzer.analyze(market_data)

    expected_prices = (
        market_data.pivot(
            index="date",
            columns="ticker",
            values="adjusted_close",
        )
        .reindex(
            columns=[
                "BTC-USD",
                "QQQ",
                "SPY",
            ]
        )
        .sort_index()
    )

    expected_returns = expected_prices.pct_change(fill_method=None).iloc[1:]

    assert isinstance(
        result,
        MultiAssetRiskResult,
    )

    assert result.observations == 4

    assert result.periods_per_year == 252

    assert result.date_min == pd.Timestamp("2025-01-02")

    assert result.date_max == pd.Timestamp("2025-01-05")

    assert result.tickers == (
        "BTC-USD",
        "QQQ",
        "SPY",
    )

    pdt.assert_frame_equal(
        result.aligned_returns,
        expected_returns,
    )

    pdt.assert_frame_equal(
        result.correlation_matrix,
        expected_returns.corr(),
    )

    pdt.assert_frame_equal(
        result.covariance_matrix,
        expected_returns.cov(),
    )

    pdt.assert_frame_equal(
        result.annualized_covariance_matrix,
        expected_returns.cov() * 252,
    )


def test_analyze_excludes_dates_not_shared_by_every_ticker() -> None:
    market_data = create_market_data()

    extra_btc_row = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-06",
                ]
            ),
            "ticker": [
                "BTC-USD",
            ],
            "adjusted_close": [
                42500.0,
            ],
        }
    )

    market_data = pd.concat(
        [
            market_data,
            extra_btc_row,
        ],
        ignore_index=True,
    )

    analyzer = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(minimum_common_observations=4),
    )

    result = analyzer.analyze(market_data)

    assert result.date_max == pd.Timestamp("2025-01-05")

    assert pd.Timestamp("2025-01-06") not in result.aligned_returns.index


def test_analyze_uses_one_complete_sample_for_every_matrix_cell() -> None:
    market_data = create_market_data()

    market_data.loc[
        (market_data["ticker"] == "QQQ")
        & (market_data["date"] == pd.Timestamp("2025-01-03")),
        "date",
    ] = pd.Timestamp("2025-01-06")

    analyzer = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(minimum_common_observations=3),
    )

    result = analyzer.analyze(market_data)

    assert result.observations == 3

    assert result.aligned_returns.notna().to_numpy().all()

    assert list(result.aligned_returns.index) == list(
        pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-04",
                "2025-01-05",
            ]
        )
    )


def test_analyze_sorts_tickers_and_dates_deterministically() -> None:
    market_data = create_market_data().sample(
        frac=1.0,
        random_state=7,
    )

    analyzer = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(minimum_common_observations=4),
    )

    result = analyzer.analyze(market_data)

    assert result.tickers == (
        "BTC-USD",
        "QQQ",
        "SPY",
    )

    assert result.aligned_returns.index.is_monotonic_increasing

    assert list(result.correlation_matrix.columns) == [
        "BTC-USD",
        "QQQ",
        "SPY",
    ]


def test_analyze_does_not_modify_input_dataframe() -> None:
    market_data = create_market_data()

    original = market_data.copy(deep=True)

    analyzer = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(minimum_common_observations=4),
    )

    analyzer.analyze(market_data)

    pdt.assert_frame_equal(
        market_data,
        original,
    )


@pytest.mark.parametrize(
    "missing_column",
    [
        "date",
        "ticker",
        "adjusted_close",
    ],
)
def test_analyze_rejects_missing_required_columns(
    missing_column: str,
) -> None:
    market_data = create_market_data().drop(columns=missing_column)

    analyzer = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(minimum_common_observations=4),
    )

    with pytest.raises(
        ValueError,
        match=("missing required columns"),
    ):
        analyzer.analyze(market_data)


@pytest.mark.parametrize(
    (
        "transform",
        "error_type",
        "message",
    ),
    [
        (
            lambda data: data.assign(date=(data["date"].astype(str))),
            TypeError,
            ("date must contain datetime values"),
        ),
        (
            lambda data: data.assign(ticker=""),
            ValueError,
            ("ticker must contain non-empty strings"),
        ),
        (
            lambda data: data.assign(adjusted_close="100.0"),
            TypeError,
            ("adjusted_close must contain numeric values"),
        ),
        (
            lambda data: data.assign(adjusted_close=np.inf),
            ValueError,
            ("adjusted_close must contain only finite values"),
        ),
        (
            lambda data: data.assign(adjusted_close=0.0),
            ValueError,
            ("adjusted_close must contain positive values"),
        ),
    ],
)
def test_analyze_rejects_invalid_market_values(
    transform: Callable[
        [pd.DataFrame],
        pd.DataFrame,
    ],
    error_type: type[Exception],
    message: str,
) -> None:
    market_data = transform(create_market_data())

    analyzer = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(minimum_common_observations=4),
    )

    with pytest.raises(
        error_type,
        match=message,
    ):
        analyzer.analyze(market_data)


def test_analyze_rejects_non_dataframe_input() -> None:
    analyzer = MultiAssetRiskAnalyzer()

    with pytest.raises(
        TypeError,
        match=("market_data must be a pandas DataFrame"),
    ):
        analyzer.analyze(
            []  # type: ignore[arg-type]
        )


def test_analyze_rejects_duplicate_ticker_date_rows() -> None:
    market_data = create_market_data()

    market_data = pd.concat(
        [
            market_data,
            market_data.iloc[[0]],
        ],
        ignore_index=True,
    )

    analyzer = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(minimum_common_observations=4),
    )

    with pytest.raises(
        ValueError,
        match=("duplicate ticker-date rows"),
    ):
        analyzer.analyze(market_data)


def test_analyze_rejects_single_ticker_dataset() -> None:
    complete_market_data = create_market_data()

    spy_mask = complete_market_data["ticker"] == "SPY"

    market_data: pd.DataFrame = complete_market_data.loc[
        spy_mask,
        :,
    ].copy()

    analyzer = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(minimum_common_observations=4),
    )

    with pytest.raises(
        ValueError,
        match=("at least two tickers"),
    ):
        analyzer.analyze(market_data)


def test_analyze_rejects_insufficient_shared_history() -> None:
    market_data = create_market_data()

    analyzer = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(minimum_common_observations=5),
    )

    with pytest.raises(
        ValueError,
        match=("at least 6 shared price dates"),
    ):
        analyzer.analyze(market_data)


def test_analyze_rejects_constant_return_series() -> None:
    market_data = create_market_data()

    market_data.loc[
        market_data["ticker"] == "SPY",
        "adjusted_close",
    ] = 100.0

    analyzer = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(minimum_common_observations=4),
    )

    with pytest.raises(
        ValueError,
        match=("correlation matrix contains undefined"),
    ):
        analyzer.analyze(market_data)


@pytest.mark.parametrize(
    (
        "create_config",
        "message",
    ),
    [
        (
            lambda: MultiAssetRiskAnalysisConfig(periods_per_year=0),
            ("periods_per_year must be a positive integer"),
        ),
        (
            lambda: MultiAssetRiskAnalysisConfig(minimum_common_observations=1),
            ("minimum_common_observations must be an integer of at least 2"),
        ),
        (
            lambda: MultiAssetRiskAnalysisConfig(price_column=""),
            ("price_column must be a non-empty string"),
        ),
    ],
)
def test_config_rejects_invalid_values(
    create_config: Callable[
        [],
        MultiAssetRiskAnalysisConfig,
    ],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        create_config()
