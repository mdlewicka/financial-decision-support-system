from datetime import UTC, datetime

from pydantic import BaseModel, Field


class AssetAllocation(BaseModel):
    """Model opisujący ile procent budżetu przypisujemy do danego aktywa."""

    asset_ticker: str = Field(..., description="Ticker aktywa, np. EURPLN=X, ^GSPC")
    weight: float = Field(
        ..., ge=0.0, le=1.0, description="Waga w portfelu (od 0 do 1)"
    )


class PortfolioState(BaseModel):
    """Model opisujący bieżący stan portfela inwestycyjnego."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_value_usd: float = Field(
        ...,
        gt=0.0,
        description="Całkowita wartość portfela w USD",
    )
    current_allocations: list[AssetAllocation] = Field(
        ...,
        description="Lista obecnych wag aktywów",
    )


class RiskProfile(BaseModel):
    """Model opisujący ograniczenia ryzyka ustawiane przez dyrektora finansowego (CFO)."""

    risk_level: str = Field(..., description="Poziom ryzyka: LOW, MEDIUM, HIGH")
    max_drawdown_tolerance: float = Field(
        ..., ge=0.0, le=0.5, description="Maksymalna akceptowalna strata (Drawdown)"
    )
