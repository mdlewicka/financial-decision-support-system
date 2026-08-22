from math import sqrt
from statistics import mean, stdev


class AnalyticsService:
    def calculate_returns(self, prices: list[float]) -> list[float]:
        if len(prices) < 2:
            return []

        returns = []

        for previous_price, current_price in zip(
            prices,
            prices[1:],
            strict=False,
        ):
            daily_return = (current_price / previous_price) - 1
            returns.append(daily_return)

        return returns

    def calculate_average_return(self, returns: list[float]) -> float:
        if not returns:
            return 0.0

        return mean(returns)

    def calculate_volatility(
        self,
        returns: list[float],
        annualize: bool = True,
    ) -> float:
        if len(returns) < 2:
            return 0.0

        volatility = stdev(returns)

        if annualize:
            volatility *= sqrt(252)

        return volatility
