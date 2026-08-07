from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentClassifyExtract(BaseModel):
    doc_type: str = Field(
        description=(
            "One of SUPERSEDED_DRAFT, LOAN_SUPERSEDED, LOAN, AUDIT_NOTES, "
            "ADJUSTMENT_SOURCE, KYC, AUDIT_PLANNING, or NOISE."
        ),
    )
    marker_quote: str = Field(
        description="Verbatim quote of the marker phrase that determined doc_type.",
    )
    acc_ids: list[str] = Field(
        default_factory=list,
        description='Account ids visible on the page (e.g. "ACC-7206").',
    )
