from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from agent.stages import s1_ingest, s2_classify

OPEN_DATA = Path(__file__).resolve().parents[1] / "data" / "open"

EXPECTED_PDF_COUNTS = {
    "NOISE": 145,
    "LOAN": 12,
    "LOAN_SUPERSEDED": 12,
    "AUDIT_NOTES": 12,
    "KYC": 11,
    "AUDIT_PLANNING": 8,
}


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("pipeline")
    s1_ingest.run(input_dir=OPEN_DATA, work_dir=work)
    s2_classify.run(work_dir=work)
    return work


def test_ingest_open_dataset_constants(work_dir: Path) -> None:
    inventory = json.loads((work_dir / "01_inventory.json").read_text(encoding="utf-8"))
    documents = inventory["documents"]

    pdf_docs = [doc for doc in documents.values() if doc["file_type"] == "pdf"]
    assert len(pdf_docs) == 200
    assert sum(doc["page_count"] for doc in pdf_docs) == 843

    scan_docs = [
        (doc_id, doc)
        for doc_id, doc in documents.items()
        if doc.get("ocr_pages")
    ]
    assert len(scan_docs) == 1
    _scan_id, scan_doc = scan_docs[0]
    assert len(scan_doc["pages"][0].strip()) < 20
    assert len(scan_doc["ocr_pages"]) == scan_doc["page_count"]
    for rel_path in scan_doc["ocr_pages"]:
        assert (work_dir / rel_path).is_file()

    assert "904dea48b34b" in documents
    assert "4a5315740e89" in documents
    assert documents["904dea48b34b"]["file_type"] == "txt"
    assert documents["4a5315740e89"]["file_type"] == "csv"
    txt_content = documents["904dea48b34b"]["pages"][0]
    assert "действующей считается" in txt_content
    assert "только текущая редакция" in txt_content

    unreadable_paths = {entry["path"] for entry in inventory["unreadable"]}
    assert any("Thumbs.db" in path for path in unreadable_paths)


def test_classify_open_dataset_counts(work_dir: Path) -> None:
    classified = json.loads((work_dir / "02_classified.json").read_text(encoding="utf-8"))
    inventory = json.loads((work_dir / "01_inventory.json").read_text(encoding="utf-8"))

    pdf_counts: Counter[str] = Counter()
    for doc_id, record in classified["documents"].items():
        if inventory["documents"][doc_id]["file_type"] != "pdf":
            continue
        pdf_counts[record["doc_type"]] += 1

    assert pdf_counts == Counter(EXPECTED_PDF_COUNTS)
    assert sum(pdf_counts.values()) == 200


def test_audit_planning_marked_not_dropped(work_dir: Path) -> None:
    classified = json.loads((work_dir / "02_classified.json").read_text(encoding="utf-8"))
    planning = [
        record
        for record in classified["documents"].values()
        if record["doc_type"] == "AUDIT_PLANNING"
    ]
    assert len(planning) == 8
    assert all(record.get("exclude_from_extraction") is True for record in planning)


def test_superseded_checked_before_loan(work_dir: Path) -> None:
    classified = json.loads((work_dir / "02_classified.json").read_text(encoding="utf-8"))
    assert (
        sum(
            1
            for record in classified["documents"].values()
            if record["doc_type"] == "LOAN_SUPERSEDED"
        )
        == 12
    )
    assert (
        sum(1 for record in classified["documents"].values() if record["doc_type"] == "LOAN")
        == 12
    )
