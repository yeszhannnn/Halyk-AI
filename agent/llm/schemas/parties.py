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
        description=(
            "Table rows beneath the dossier header: counterparty name and percentage "
            "(voting-rights share or pledged-asset share, depending on table_semantics)."
        ),
    )
    table_semantics: str = Field(
        description=(
            "RELATED_PARTY when the rule sentence defines related parties by ownership "
            "threshold (>= threshold is related); UNRESTRICTED_SUBSIDIARY when the rule "
            "sentence defines a security perimeter and subsidiaries below the threshold "
            "are unrestricted."
        ),
    )
    table_semantics_quote: str = Field(
        description="Verbatim quote of the rule sentence beneath the table.",
    )
    threshold_pct: Decimal = Field(
        description=(
            "Numeric percentage from the rule sentence beneath the table "
            "(e.g. 35.0 for related-party threshold or 50.0 for security perimeter)."
        ),
    )
    threshold_quote: str = Field(
        description="Verbatim quote of the threshold sentence (may match table_semantics_quote).",
    )
