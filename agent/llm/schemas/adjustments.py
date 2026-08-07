from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field


class EbitdaAddbackRowExtract(BaseModel):
    item: str = Field(description="One-off expense item label from the table.")
    item_quote: str = Field(description="Verbatim quote containing the item label.")
    counterparty: str = Field(description="Counterparty name from the table row.")
    counterparty_quote: str = Field(description="Verbatim quote containing the counterparty name.")
    amount: Decimal = Field(description="Row amount in USD.")
    amount_quote: str = Field(description="Verbatim quote containing the row amount.")
    included_in_addback: bool | None = Field(
        default=None,
        description=(
            "Optional model indication of whether this row is above the materiality floor. "
            "Leave null when not stated."
        ),
    )


class AdjustmentExtract(BaseModel):
    kind: str = Field(
        description=(
            "One of RECLASS, CUTOFF, EXCLUDE, OFF_LEDGER, AMOUNT_FILL, FX, "
            "EBITDA_ADDBACK, NONE, or UNRECOGNISED if the segment cannot be classified."
        ),
    )
    kind_quote: str = Field(
        description="Verbatim quote supporting the chosen kind.",
    )
    txn_id: str | None = Field(
        default=None,
        description="Ledger transaction id when explicitly stated (e.g. TXN-P1-0045).",
    )
    txn_id_quote: str = Field(default="", description="Verbatim quote containing txn_id, if any.")
    amount: Decimal | None = Field(
        default=None,
        description="Primary USD amount for the adjustment, when stated.",
    )
    amount_quote: str = Field(default="", description="Verbatim quote containing the primary amount.")
    counterparty: str | None = Field(default=None, description="Counterparty name when stated.")
    counterparty_quote: str = Field(default="", description="Verbatim quote containing counterparty.")
    from_category: str | None = Field(
        default=None,
        description="Original category for RECLASS (short English slug, e.g. consulting).",
    )
    from_category_quote: str = Field(default="")
    to_category: str | None = Field(
        default=None,
        description="Target category for RECLASS (short English slug, e.g. opex, interest).",
    )
    to_category_quote: str = Field(default="")
    category: str | None = Field(
        default=None,
        description="Category for OFF_LEDGER synthetic rows (short English slug).",
    )
    category_quote: str = Field(default="")
    fx_source_amount: Decimal | None = Field(
        default=None,
        description="Foreign-currency invoice amount for FX disclosures.",
    )
    fx_source_currency: str | None = Field(
        default=None,
        description="ISO-like currency code for the foreign invoice (e.g. EUR).",
    )
    fx_settlement_usd: Decimal | None = Field(
        default=None,
        description="USD settlement amount for FX disclosures.",
    )
    fx_quote: str = Field(default="", description="Verbatim quote covering both FX amounts.")
    materiality_floor: Decimal | None = Field(
        default=None,
        description="Materiality threshold for EBITDA_ADDBACK tables.",
    )
    materiality_floor_quote: str = Field(default="")
    ebitda_rows: list[EbitdaAddbackRowExtract] = Field(
        default_factory=list,
        description="Table rows when kind is EBITDA_ADDBACK.",
    )


class VisionAdjustmentsExtract(BaseModel):
    items: list[AdjustmentExtract] = Field(
        default_factory=list,
        description="Zero or more adjustments visible on the scanned page(s).",
    )
