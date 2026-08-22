from math import sqrt
from statistics import covariance

from backend.src.services.analytics_service import AnalyticsService
from backend.src.services.market_data_service import MarketDataService


class PortfolioAnalyticsService:
    def __init__(
        self,
        market_data_service: MarketDataService,
        analytics_service: AnalyticsService,
    ):
        self.market_data_service = market_data_service
        self.analytics_service = analytics_service

    def calculate_portfolio_return(
        self,
        asset_returns: list[float],
        weights: list[float],
    ) -> float:
        if not asset_returns:
            raise ValueError("Lista stóp zwrotu nie może być pusta.")

        if len(asset_returns) != len(weights):
            raise ValueError("Liczba stóp zwrotu musi być równa liczbie wag.")

        self._validate_weights(weights)

        return sum(
            asset_return * weight
            for asset_return, weight in zip(
                asset_returns,
                weights,
                strict=True,
            )
        )

    def calculate_covariance_matrix(
        self,
        asset_returns: list[list[float]],
    ) -> list[list[float]]:
        if not asset_returns:
            raise ValueError("Lista stóp zwrotu aktywów nie może być pusta.")

        observations_count = len(asset_returns[0])

        if observations_count < 2:
            raise ValueError(
                "Każde aktywo musi mieć co najmniej dwie obserwacje stóp zwrotu."
            )

        for returns in asset_returns:
            if len(returns) != observations_count:
                raise ValueError("Każde aktywo musi mieć tę samą liczbę obserwacji.")

        covariance_matrix = []

        for first_asset_returns in asset_returns:
            row = []

            for second_asset_returns in asset_returns:
                covariance_value = covariance(
                    first_asset_returns,
                    second_asset_returns,
                )

                row.append(covariance_value)

            covariance_matrix.append(row)

        return covariance_matrix

    def calculate_portfolio_volatility(
        self,
        weights: list[float],
        covariance_matrix: list[list[float]],
        annualize: bool = True,
    ) -> float:
        if not weights:
            raise ValueError("Lista wag nie może być pusta.")

        self._validate_weights(weights)

        number_of_assets = len(weights)

        if len(covariance_matrix) != number_of_assets:
            raise ValueError(
                "Macierz kowariancji musi mieć tyle wierszy, ile portfel ma aktywów."
            )

        for row in covariance_matrix:
            if len(row) != number_of_assets:
                raise ValueError("Macierz kowariancji musi być kwadratowa.")

        portfolio_variance = 0.0

        for row_index in range(number_of_assets):
            for column_index in range(number_of_assets):
                portfolio_variance += (
                    weights[row_index]
                    * weights[column_index]
                    * covariance_matrix[row_index][column_index]
                )

        if portfolio_variance < 0:
            raise ValueError("Wariancja portfela nie może być ujemna.")

        portfolio_volatility = sqrt(portfolio_variance)

        if annualize:
            portfolio_volatility *= sqrt(252)

        return portfolio_volatility

    def _load_market_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, dict[str, float]]:
        rates_by_symbol = {}

        for symbol in symbols:
            historical_data = self.market_data_service.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

            if not historical_data:
                raise ValueError(f"Brak danych historycznych dla {symbol}.")

            rates_by_symbol[symbol] = {
                item["date"]: item["rate"] for item in historical_data
            }

        return rates_by_symbol

    def analyze_market_portfolio(
        self,
        symbols: list[str],
        weights: list[float],
        start_date: str,
        end_date: str,
    ) -> dict:
        if not symbols:
            raise ValueError("Lista instrumentów nie może być pusta.")

        if len(symbols) != len(weights):
            raise ValueError("Liczba instrumentów musi być równa liczbie wag.")

        self._validate_weights(weights)

        rates_by_symbol = self._load_market_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

        common_dates = set.intersection(
            *[set(symbol_rates.keys()) for symbol_rates in rates_by_symbol.values()]
        )

        sorted_common_dates = sorted(common_dates)

        if len(sorted_common_dates) < 3:
            raise ValueError("Za mało wspólnych obserwacji dla wybranych aktywów.")

        returns_by_symbol = {}
        asset_returns = []
        average_daily_returns = []

        for symbol in symbols:
            prices = [rates_by_symbol[symbol][date] for date in sorted_common_dates]

            returns = self.analytics_service.calculate_returns(prices)

            average_return = self.analytics_service.calculate_average_return(returns)

            returns_by_symbol[symbol] = returns
            asset_returns.append(returns)
            average_daily_returns.append(average_return)

        covariance_matrix = self.calculate_covariance_matrix(asset_returns)

        portfolio_average_daily_return = self.calculate_portfolio_return(
            asset_returns=average_daily_returns,
            weights=weights,
        )

        annualized_expected_return = portfolio_average_daily_return * 252

        annualized_volatility = self.calculate_portfolio_volatility(
            weights=weights,
            covariance_matrix=covariance_matrix,
            annualize=True,
        )

        return {
            "symbols": symbols,
            "weights": weights,
            "start_date": start_date,
            "end_date": end_date,
            "common_price_observations": len(sorted_common_dates),
            "return_observations": len(sorted_common_dates) - 1,
            "average_daily_returns": {
                symbol: average_return
                for symbol, average_return in zip(
                    symbols,
                    average_daily_returns,
                    strict=True,
                )
            },
            "portfolio_average_daily_return": (portfolio_average_daily_return),
            "annualized_expected_return": (annualized_expected_return),
            "annualized_volatility": annualized_volatility,
            "covariance_matrix": covariance_matrix,
        }

    def _validate_weights(
        self,
        weights: list[float],
    ) -> None:
        if not weights:
            raise ValueError("Lista wag nie może być pusta.")

        if any(weight < 0 for weight in weights):
            raise ValueError("Wagi nie mogą być ujemne.")

        if abs(sum(weights) - 1.0) > 0.0001:
            raise ValueError("Suma wag portfela musi wynosić 1.")
