import pandas as pd

from backend.src.services.exceptions import MarketDataStandardizationError


class MarketDataStandardizer:
    """Przekształca surowe dane rynkowe do wspólnego formatu systemu."""

    _COLUMN_MAPPING = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    }

    _OUTPUT_COLUMNS = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]

    def standardize(
        self,
        data: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Standaryzuje dane historyczne pojedynczego instrumentu."""

        normalized_ticker = ticker.strip().upper()

        if not normalized_ticker:
            raise MarketDataStandardizationError("Ticker nie może być pusty.")

        if data is None or data.empty:
            raise MarketDataStandardizationError(
                f"Nie można standaryzować pustych danych dla {normalized_ticker!r}."
            )

        if isinstance(data.columns, pd.MultiIndex):
            raise MarketDataStandardizationError(
                "Dane wejściowe zawierają kolumny MultiIndex. "
                "Standardizer oczekuje danych dla pojedynczego tickera."
            )

        self._validate_required_columns(data)

        standardized = pd.DataFrame()

        standardized["date"] = self._standardize_dates(data.index)
        standardized["ticker"] = normalized_ticker

        for source_column, target_column in self._COLUMN_MAPPING.items():
            standardized[target_column] = pd.to_numeric(
                data[source_column],
                errors="coerce",
            ).to_numpy()

        standardized = standardized.sort_values(
            by="date",
            kind="stable",
        ).reset_index(drop=True)

        return standardized[self._OUTPUT_COLUMNS].copy()

    def _validate_required_columns(self, data: pd.DataFrame) -> None:
        missing_columns = [
            column for column in self._COLUMN_MAPPING if column not in data.columns
        ]

        if missing_columns:
            missing = ", ".join(missing_columns)

            raise MarketDataStandardizationError(
                f"Brakuje wymaganych kolumn: {missing}."
            )

    @staticmethod
    def _standardize_dates(index: pd.Index) -> pd.Series:
        dates = pd.to_datetime(
            index,
            errors="coerce",
            utc=True,
        )

        dates = dates.tz_convert(None)

        return pd.Series(dates, name="date")
