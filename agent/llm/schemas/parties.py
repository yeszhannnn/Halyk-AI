from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field


class OwnershipRowExtract(BaseModel):
    counterparty: str = Field(description="Legal name of the counterparty organisation.")
    counterparty_quote: str = Field(
        description="Verbatim quote containing the counterparty name from the dossier.",
    )
    ownership_pct: Decimal = Field(
        description="Group share of voting rights as a percentage (e.g. 41.2 for 41.2%).",
    )
    ownership_pct_quote: str = Field(
        description="Verbatim quote containing the ownership percentage.",
    )


class KycPartiesExtract(BaseModel):
    header_account: str = Field(
        description="Account id from the dossier header line (e.g. ACC-7805).",
    )
    header_account_quote: str = Field(
        description="Verbatim quote containing the header account id.",
    )
    ownership_rows: list[OwnershipRowExtract] = Field(
        description="Ownership table rows: counterparty name and voting-rights percentage.",
    )
    threshold_pct: Decimal = Field(
        description=(
            "Related-party threshold percentage from the dossier sentence "
            "(e.g. 35.0 when text says '35.0% and more')."
        ),
    )
    threshold_quote: str = Field(
        description="Verbatim quote of the related-party threshold sentence.",
    )
