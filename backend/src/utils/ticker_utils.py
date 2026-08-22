from collections.abc import Iterable


def normalize_tickers(
    tickers: Iterable[str],
) -> tuple[str, ...]:
    normalized_tickers = tuple(
        dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip())
    )

    if not normalized_tickers:
        raise ValueError("Lista tickerów nie może być pusta.")

    return normalized_tickers
