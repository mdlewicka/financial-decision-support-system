from datetime import date
from pathlib import Path

from backend.src.domain.assets import Asset, AssetClass, MarketRegion

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_MARKET_DATA_DIR = RAW_DATA_DIR / "market"
INTERIM_MARKET_DATA_DIR = INTERIM_DATA_DIR / "market"
PROCESSED_MARKET_DATA_DIR = PROCESSED_DATA_DIR / "market"

DATASETS_DIR = DATA_DIR / "datasets"

MARKET_DATASETS_DIR = DATASETS_DIR / "market"

TRAIN_DATASET_FILENAME = "train.csv"

VALIDATION_DATASET_FILENAME = "validation.csv"

TEST_DATASET_FILENAME = "test.csv"

COMBINED_MARKET_DATA_FILENAME = "market_data.csv"
MARKET_FEATURES_FILENAME = "market_features.csv"

MARKET_DATA_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)

MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)

HISTORY_START_DATE = date(2010, 1, 1)
MINIMUM_MODEL_OBSERVATIONS_PER_TICKER = 1000
DATA_INTERVAL = "1d"

RAW_DATA_FILE_FORMAT = "csv"
DATE_COLUMN_NAME = "date"


SUPPORTED_ASSETS: dict[str, Asset] = {
    "epol": Asset(
        asset_id="epol",
        name="iShares MSCI Poland ETF",
        ticker="EPOL",
        asset_class=AssetClass.EQUITY,
        region=MarketRegion.POLAND,
        quote_currency="USD",
        description="ETF reprezentujący polski rynek akcji.",
    ),
    "vgk": Asset(
        asset_id="vgk",
        name="Vanguard FTSE Europe ETF",
        ticker="VGK",
        asset_class=AssetClass.EQUITY,
        region=MarketRegion.EUROPE,
        quote_currency="USD",
        description="ETF reprezentujący rozwinięte europejskie rynki akcji.",
    ),
    "spy": Asset(
        asset_id="spy",
        name="SPDR S&P 500 ETF Trust",
        ticker="SPY",
        asset_class=AssetClass.EQUITY,
        region=MarketRegion.UNITED_STATES,
        quote_currency="USD",
        description="ETF reprezentujący największe spółki amerykańskie.",
    ),
    "vt": Asset(
        asset_id="vt",
        name="Vanguard Total World Stock ETF",
        ticker="VT",
        asset_class=AssetClass.EQUITY,
        region=MarketRegion.GLOBAL,
        quote_currency="USD",
        description="ETF reprezentujący globalny rynek akcji.",
    ),
    "qqq": Asset(
        asset_id="qqq",
        name="Invesco QQQ Trust",
        ticker="QQQ",
        asset_class=AssetClass.EQUITY,
        region=MarketRegion.UNITED_STATES,
        quote_currency="USD",
        description=(
            "ETF reprezentujący spółki notowane w indeksie Nasdaq-100, "
            "z silną ekspozycją na sektor technologiczny."
        ),
    ),
    "gld": Asset(
        asset_id="gld",
        name="SPDR Gold Shares",
        ticker="GLD",
        asset_class=AssetClass.COMMODITY,
        region=MarketRegion.GLOBAL,
        quote_currency="USD",
        description="ETF zapewniający ekspozycję na cenę złota.",
    ),
    "btc_usd": Asset(
        asset_id="btc_usd",
        name="Bitcoin / US Dollar",
        ticker="BTC-USD",
        asset_class=AssetClass.CRYPTOCURRENCY,
        region=MarketRegion.GLOBAL,
        quote_currency="USD",
        description="Cena Bitcoina wyrażona w dolarach amerykańskich.",
    ),
    "agg": Asset(
        asset_id="agg",
        name="iShares Core U.S. Aggregate Bond ETF",
        ticker="AGG",
        asset_class=AssetClass.BOND,
        region=MarketRegion.UNITED_STATES,
        quote_currency="USD",
        description="ETF reprezentujący szeroki rynek obligacji amerykańskich.",
    ),
}


SUPPORTED_TICKERS: tuple[str, ...] = tuple(
    asset.ticker for asset in SUPPORTED_ASSETS.values()
)
