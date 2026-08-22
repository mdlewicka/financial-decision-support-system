from abc import ABC, abstractmethod
from typing import Any


class MarketDataProvider(ABC):
    """
    Wspólny interfejs dla providerów danych rynkowych.
    """

    @abstractmethod
    def get_latest_price(self, symbol: str) -> dict[str, Any]:
        """
        Pobiera najnowszą dostępną cenę instrumentu.
        """
        pass

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """
        Pobiera dane historyczne dla instrumentu.
        """
        pass
