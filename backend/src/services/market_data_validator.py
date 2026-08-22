import pandas as pd

from backend.src.services.exceptions import MarketDataValidationError


class MarketDataValidator:
    """Sprawdza jakość i spójność wystandaryzowanych danych rynkowych."""

    _REQUIRED_COLUMNS = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]

    _PRICE_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
    ]

    def __init__(self, minimum_observations: int = 2) -> None:
        if minimum_observations <= 0:
            raise ValueError("Minimalna liczba obserwacji musi być większa od zera.")

        self._minimum_observations = minimum_observations

    def validate(
        self,
        data: pd.DataFrame,
        expected_ticker: str | None = None,
    ) -> pd.DataFrame:
        """Sprawdza, czy dane spełniają wymagania jakościowe systemu."""

        self._validate_not_empty(data)
        self._validate_required_columns(data)
        self._validate_minimum_observations(data)
        self._validate_missing_values(data)
        self._validate_dates(data)
        self._validate_tickers(data, expected_ticker)
        self._validate_numeric_columns(data)
        self._validate_price_values(data)
        self._validate_volume(data)
        self._validate_ohlc_relationships(data)

        return data

    @staticmethod
    def _validate_not_empty(data: pd.DataFrame) -> None:
        if data is None or data.empty:
            raise MarketDataValidationError(
                "Nie można walidować pustych danych rynkowych."
            )

    def _validate_required_columns(self, data: pd.DataFrame) -> None:
        missing_columns = [
            column for column in self._REQUIRED_COLUMNS if column not in data.columns
        ]

        if missing_columns:
            missing = ", ".join(missing_columns)

            raise MarketDataValidationError(f"Brakuje wymaganych kolumn: {missing}.")

    def _validate_minimum_observations(
        self,
        data: pd.DataFrame,
    ) -> None:
        observations_count = len(data)

        if observations_count < self._minimum_observations:
            raise MarketDataValidationError(
                "Liczba obserwacji jest zbyt mała. "
                f"Otrzymano {observations_count}, "
                f"wymagane minimum: {self._minimum_observations}."
            )

    def _validate_missing_values(self, data: pd.DataFrame) -> None:
        missing_counts = data[self._REQUIRED_COLUMNS].isna().sum()
        columns_with_missing_values = missing_counts[missing_counts > 0]

        if not columns_with_missing_values.empty:
            details = ", ".join(
                f"{column}: {count}"
                for column, count in columns_with_missing_values.items()
            )

            raise MarketDataValidationError(f"Wykryto brakujące wartości: {details}.")

    @staticmethod
    def _validate_dates(data: pd.DataFrame) -> None:
        if not pd.api.types.is_datetime64_any_dtype(data["date"]):
            raise MarketDataValidationError("Kolumna 'date' musi mieć typ datetime.")

        if data["date"].duplicated().any():
            duplicated_dates = (
                data.loc[data["date"].duplicated(keep=False), "date"]
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            )

            raise MarketDataValidationError(
                f"Wykryto zduplikowane daty: {', '.join(duplicated_dates)}."
            )

        if not data["date"].is_monotonic_increasing:
            raise MarketDataValidationError(
                "Dane nie są uporządkowane chronologicznie."
            )

    @staticmethod
    def _validate_tickers(
        data: pd.DataFrame,
        expected_ticker: str | None,
    ) -> None:
        unique_tickers = data["ticker"].astype(str).unique().tolist()

        if len(unique_tickers) != 1:
            tickers = ", ".join(unique_tickers)

            raise MarketDataValidationError(
                f"Dane muszą dotyczyć dokładnie jednego tickera. Wykryto: {tickers}."
            )

        actual_ticker = unique_tickers[0].strip().upper()

        if not actual_ticker:
            raise MarketDataValidationError("Ticker w danych nie może być pusty.")

        if expected_ticker is None:
            return

        normalized_expected_ticker = expected_ticker.strip().upper()

        if not normalized_expected_ticker:
            raise MarketDataValidationError("Oczekiwany ticker nie może być pusty.")

        if actual_ticker != normalized_expected_ticker:
            raise MarketDataValidationError(
                "Ticker w danych jest niezgodny z oczekiwanym tickerem. "
                f"Otrzymano {actual_ticker!r}, "
                f"oczekiwano {normalized_expected_ticker!r}."
            )

    def _validate_numeric_columns(self, data: pd.DataFrame) -> None:
        numeric_columns = [*self._PRICE_COLUMNS, "volume"]

        invalid_columns = [
            column
            for column in numeric_columns
            if not pd.api.types.is_numeric_dtype(data[column])
        ]

        if invalid_columns:
            invalid = ", ".join(invalid_columns)

            raise MarketDataValidationError(
                f"Kolumny muszą mieć typ numeryczny: {invalid}."
            )

    def _validate_price_values(self, data: pd.DataFrame) -> None:
        invalid_price_counts = (data[self._PRICE_COLUMNS] <= 0).sum()

        invalid_columns = invalid_price_counts[invalid_price_counts > 0]

        if not invalid_columns.empty:
            details = ", ".join(
                f"{column}: {count}" for column, count in invalid_columns.items()
            )

            raise MarketDataValidationError(
                f"Ceny muszą być większe od zera. Niepoprawne wartości: {details}."
            )

    @staticmethod
    def _validate_volume(data: pd.DataFrame) -> None:
        invalid_volume_count = int((data["volume"] < 0).sum())

        if invalid_volume_count > 0:
            raise MarketDataValidationError(
                "Wolumen nie może być ujemny. "
                f"Liczba niepoprawnych obserwacji: "
                f"{invalid_volume_count}."
            )

    @staticmethod
    def _validate_ohlc_relationships(
        data: pd.DataFrame,
    ) -> None:
        invalid_low = data["low"] > data[["open", "high", "close"]].min(axis=1)

        invalid_high = data["high"] < data[["open", "low", "close"]].max(axis=1)

        invalid_rows = invalid_low | invalid_high
        invalid_rows_count = int(invalid_rows.sum())

        if invalid_rows_count > 0:
            invalid_dates = (
                data.loc[invalid_rows, "date"].dt.strftime("%Y-%m-%d").tolist()
            )

            preview = ", ".join(invalid_dates[:5])

            if invalid_rows_count > 5:
                preview = f"{preview}, ..."

            raise MarketDataValidationError(
                "Wykryto niespójne relacje cen OHLC. "
                f"Liczba niepoprawnych obserwacji: "
                f"{invalid_rows_count}. "
                f"Daty: {preview}."
            )
