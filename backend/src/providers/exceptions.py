class MarketDataProviderError(RuntimeError):
    """Bazowy wyjątek zgłaszany przez providerów danych rynkowych."""


class MarketDataDownloadError(MarketDataProviderError):
    """Błąd pobierania danych rynkowych z zewnętrznego źródła."""


class EmptyMarketDataError(MarketDataProviderError):
    """Błąd zgłaszany, gdy provider nie zwrócił żadnych obserwacji."""
