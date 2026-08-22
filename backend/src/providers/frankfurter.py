from typing import Any

import requests

from backend.src.providers.base import MarketDataProvider


class FrankfurterProvider(MarketDataProvider):
    """
    Provider pobierający kursy walut z Frankfurter API.
    """

    BASE_URL = "https://api.frankfurter.dev/v2"

    def get_latest_price(self, symbol: str) -> dict[str, Any]:
        """
        Pobiera najnowszy dostępny kurs dla pary walutowej.

        Przykład symbolu:
        EUR/USD
        EUR/CHF
        EUR/PLN
        """

        base_currency, quote_currency = self._split_symbol(symbol)

        url = f"{self.BASE_URL}/rate/{base_currency}/{quote_currency}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        return response.json()

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """
        Pobiera historyczne kursy walut z wybranego przedziału dat.

        Daty powinny mieć format:
        YYYY-MM-DD
        """

        base_currency, quote_currency = self._split_symbol(symbol)

        url = f"{self.BASE_URL}/rates"

        params = {
            "base": base_currency,
            "quotes": quote_currency,
            "from": start_date,
            "to": end_date,
        }

        response = requests.get(
            url,
            params=params,
            timeout=10,
        )
        response.raise_for_status()

        return response.json()

    @staticmethod
    def _split_symbol(symbol: str) -> tuple[str, str]:
        """
        Rozdziela symbol pary walutowej na walutę bazową i kwotowaną.

        EUR/USD:
        base_currency = EUR
        quote_currency = USD
        """

        parts = symbol.upper().split("/")

        if len(parts) != 2:
            raise ValueError(
                "Symbol pary walutowej musi mieć format BASE/QUOTE, np. EUR/USD."
            )

        base_currency, quote_currency = parts

        if len(base_currency) != 3 or len(quote_currency) != 3:
            raise ValueError("Każdy kod waluty musi składać się z trzech znaków.")

        return base_currency, quote_currency
