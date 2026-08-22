from backend.src.forecast.forecast_metrics import ForecastMetrics


def test_forecast_metrics_stores_metric_values() -> None:
    metrics = ForecastMetrics(
        observations=100,
        mae=0.01,
        mse=0.0002,
        rmse=0.014,
        r2=0.15,
        directional_accuracy=0.60,
    )

    assert metrics.observations == 100
    assert metrics.mae == 0.01
    assert metrics.mse == 0.0002
    assert metrics.rmse == 0.014
    assert metrics.r2 == 0.15
    assert metrics.directional_accuracy == 0.60
