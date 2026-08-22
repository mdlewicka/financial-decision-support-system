from dataclasses import dataclass
from enum import StrEnum


class AssetClass(StrEnum):
    """Klasy aktywów obsługiwane przez system wspomagania decyzji finansowych."""

    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CRYPTOCURRENCY = "cryptocurrency"


class MarketRegion(StrEnum):
    """Region gospodarczy reprezentowany przez dane aktywo."""

    POLAND = "poland"
    EUROPE = "europe"
    UNITED_STATES = "united_states"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class Asset:
    """Reprezentuje instrument finansowy obsługiwany przez system."""

    asset_id: str
    name: str
    ticker: str
    asset_class: AssetClass
    region: MarketRegion
    quote_currency: str
    description: str
