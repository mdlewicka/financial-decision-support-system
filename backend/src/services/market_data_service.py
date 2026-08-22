from backend.src.providers.base import MarketDataProvider
from backend.src.providers.frankfurter import FrankfurterProvider


class MarketDataService:
    """
    Serwis odpowiedzialny za pobieranie danych rynkowych.
    """

    def __init__(
        self,
        provider: MarketDataProvider | None = None,
    ) -> None:
        if provider is None:
            provider = FrankfurterProvider()

        self.provider = provider

    def get_latest_price(self, symbol: str) -> dict:
        """
        Pobiera najnowszy kurs dla wybranego instrumentu.
        """

        return self.provider.get_latest_price(symbol)

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """
        Pobiera dane historyczne dla wybranego instrumentu.
        """

        return self.provider.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
