from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from backend.src.services.market_feature_processing_service import (
    MarketFeatureProcessingService,
)


def test_process_runs_entire_pipeline() -> None:
    reader = Mock()
    feature_engineer = Mock()
    cleaner = Mock()
    writer = Mock()

    input_data = pd.DataFrame(
        {
            "ticker": ["SPY"],
        }
    )

    featured_data = pd.DataFrame(
        {
            "ticker": ["SPY"],
            "daily_return": [0.01],
        }
    )

    cleaned_data = pd.DataFrame(
        {
            "ticker": ["SPY"],
            "daily_return": [0.01],
        }
    )

    output_path = Path("data/processed/market/market_features.csv")

    reader.read.return_value = input_data

    feature_engineer.engineer.return_value = featured_data

    cleaner.clean.return_value = cleaned_data

    writer.write.return_value = output_path

    service = MarketFeatureProcessingService(
        reader=reader,
        feature_engineer=feature_engineer,
        cleaner=cleaner,
        writer=writer,
    )

    result = service.process()

    reader.read.assert_called_once_with()

    feature_engineer.engineer.assert_called_once_with(input_data)

    cleaner.clean.assert_called_once_with(featured_data)

    writer.write.assert_called_once_with(cleaned_data)

    assert result == output_path


def test_process_passes_data_between_pipeline_steps() -> None:
    reader = Mock()
    feature_engineer = Mock()
    cleaner = Mock()
    writer = Mock()

    input_data = pd.DataFrame(
        {
            "ticker": ["QQQ"],
        }
    )

    featured_data = pd.DataFrame(
        {
            "ticker": ["QQQ"],
            "daily_return": [0.02],
        }
    )

    cleaned_data = pd.DataFrame(
        {
            "ticker": ["QQQ"],
            "daily_return": [0.02],
        }
    )

    reader.read.return_value = input_data

    feature_engineer.engineer.return_value = featured_data

    cleaner.clean.return_value = cleaned_data

    writer.write.return_value = Path("market_features.csv")

    service = MarketFeatureProcessingService(
        reader=reader,
        feature_engineer=feature_engineer,
        cleaner=cleaner,
        writer=writer,
    )

    service.process()

    assert feature_engineer.engineer.call_args.args[0] is input_data

    assert cleaner.clean.call_args.args[0] is featured_data

    assert writer.write.call_args.args[0] is cleaned_data
