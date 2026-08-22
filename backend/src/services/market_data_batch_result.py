from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MarketDataDownloadFailure:
    """Opis błędu pobierania danych dla pojedynczego tickera."""

    ticker: str
    error_type: str
    message: str


@dataclass(frozen=True)
class MarketDataBatchResult:
    """Zbiorczy wynik pobierania danych dla wielu tickerów."""

    successful: dict[str, Path] = field(default_factory=dict)
    failed: dict[str, MarketDataDownloadFailure] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Informuje, czy wszystkie operacje zakończyły się sukcesem."""

        return not self.failed

    @property
    def total_count(self) -> int:
        """Zwraca liczbę wszystkich przetworzonych tickerów."""

        return len(self.successful) + len(self.failed)

    @property
    def success_count(self) -> int:
        """Zwraca liczbę poprawnie pobranych tickerów."""

        return len(self.successful)

    @property
    def failure_count(self) -> int:
        """Zwraca liczbę tickerów zakończonych błędem."""

        return len(self.failed)
