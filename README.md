# Covenant Agent

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Halyk AI Challenge](https://img.shields.io/badge/Halyk%20AI%20Challenge-2026-lightgrey)

An autonomous agent that reads an archive of corporate loan documents together with a
transaction ledger and decides, for every covenant of every borrower, whether it is met.

* **The model extracts; the code decides.** The language model returns verbatim spans
  and nothing else. Every comparison, aggregation and ratio is computed in Python over
  `Decimal`. No verdict passes through a model.
* **Three reading modalities, detected per page.** Text layers, image-rendered pages
  inside otherwise textual documents, and fully scanned files each take a different
  path. Detecting this per document rather than per page loses content silently.
* **Nothing fails quietly.** Every stage contains failures at the level of a single
  clause or document. One unreadable covenant costs one cell, never the run.

---

## Quick start

```bash
git clone <repo-url> && cd covenant-agent
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env                                  # add your API key
```

The input directory must contain `documents/`, the transaction ledger CSV, and
`submission_template.json`.

**Inspect the dataset before spending anything.** Preflight runs stages 1 to 3 only and
makes no API calls:

```bash
python -m agent preflight --input data/private
```

It prints the shape of the archive — scenario count, slot names, document type counts,
image pages, ledger rows per borrower, and any borrower missing a contract or dossier.
Departures from the expected shape are warnings, never errors.

**Then the run.** One command produces the submission:

```bash
python -m agent --input data/private --out submission.json
```

There is no manual step anywhere in the pipeline and no path through the code where an
answer can be entered by hand.

**Provider is a single environment variable.** `LLM_PROVIDER=openai` or
`LLM_PROVIDER=anthropic`; both paths are exercised and switching requires no code
change.

Stages skip when their artifact already exists, so a run resumes rather than restarts.
Deleting `04a_covenants.json` re-extracts covenants without repeating the two minutes of
PDF parsing.

```bash
python eval/score.py submission.json eval/ground_truth.json    # score against a key
python eval/diagnose.py submission.json --cell P5/6.1          # why one cell is wrong
```

`diagnose --cell` prints every ledger row that entered each leg with the reason it was
included, the derived terms, the substituted expression and the comparison. It is the
difference between "this number is wrong" and "this category matched two rows it
should not have".

---

## How it works

Seven stages, each reading one artifact and writing the next.

| Stage | Does | Writes |
|---|---|---|
| 1 · Ingest | text per page, per-page image detection, page rendering | `01_inventory.json` |
| 2 · Classify | document type from content markers; filenames are opaque hashes | `02_classified.json` |
| 3 · Bind | borrower ↔ documents, resolved through the ledger, never by company name | `03_bound.json` |
| 4a · Covenants | thresholds, comparison direction, metric shapes, springing conditions | `04a_covenants.json` |
| 4b · Classification tables | ownership and pledged-asset tables with the rule stated beneath each | `04b_parties.json` |
| 4c · Adjustments | eight kinds of auditor adjustment, from text and from image pages | `04c_adjustments.json` |
| 5 · Ledger | filter by account, apply adjustments in order, convert currency | `05_ledger.parquet` |
| 6 · Evaluate | applicability gate, full-precision comparison, counterfactual evidence | `06_evaluated.json` |
| 7 · Emit | evidence trace, then a pure projection into the submission template | `trace.json`, `submission.json` |

`submission.json` is a pure function of `trace.json`. A change to output formatting is a
one-function edit and a one-second rebuild, not another pipeline run.

---

## Architecture

### Document layer

`pdfplumber` for text and word-level bounding boxes, `PyMuPDF` for page rendering and
for `search_for`.

The pairing matters more than either library. The model is asked to return the exact
printed wording of a threshold; the code then locates that string on the page with
`search_for` and derives the coordinates itself. Coordinates are therefore never
hallucinated, because the model is never asked for them. A returned span that is not a
substring of its page is a hallucination by construction and is rejected before it
reaches the ledger.

### No retrieval layer, deliberately

Two hundred documents is not a retrieval problem. Document type is decided by exact
marker match against the text, which is deterministic and auditable — a classification
can be justified by pointing at the phrase that produced it.

An embedding index would replace an exact decision with a similarity score, and
similarity is precisely the wrong tool where a superseded contract and its operative
replacement differ by one header line. It would also introduce run-to-run variation in
the one place where variation is most expensive: which document a borrower is bound to.

### Vision, only where the text layer ends

Not a general OCR problem, and not solved with a general OCR engine. The content that
lives on image pages is tabular — ownership percentages, materiality tables — where
layout carries meaning that character recognition alone discards. Those pages are
rendered at 150 DPI and read by the same vision model under the same Pydantic schema
used for text, so the downstream code cannot tell the two apart.

Detection is per page, not per document: a five-page audit file can have two thousand
characters on page one and two image-only pages in the middle. Gating on the first page
finds fully scanned files and misses everything else.

Quote verification cannot apply to an image, so fields extracted this way are marked
`source_kind="ocr"`, carry reduced confidence, and appear in the review list.

### Extraction layer

`instructor` with Pydantic models. The schema is the contract: the model fills a typed
object or the call fails and retries.

Two rules keep the schema stable under pressure. Fields the model does not reliably
produce — quote strings, optional notes — are optional with defaults, because a missing
quote carries no score and must never block an extraction. Fields that decide an answer
— thresholds, ownership percentages — are required, because a silently defaulted
threshold produces a confident wrong verdict, which is worse than a loud failure.

Sentinel values are treated as absence, not as parse errors. A model that answers
`<UNKNOWN>` for a figure it cannot find is behaving correctly; the pipeline records the
gap and degrades rather than retrying an answer that will not change.

### Ledger layer

`pandas` for loading, `duckdb` for querying. Adjustments apply in a fixed order, because
the order changes the result: amounts are filled into existing rows first, then rows are
excluded, then reclassified, then currency-converted, and only then are off-ledger
disclosures appended as synthetic rows. Filling amounts last would leave a row invisible
to every aggregate that ran before it.

Every monetary quantity is `Decimal` end to end. Category membership is decided per
covenant from that covenant's own definition, not from a global taxonomy — the same row
counts toward different line items for different borrowers, because the contracts define
the terms differently.

### Determinism and cost

`temperature=0`, fixed seed, pinned model id. Every request is cached on disk under a
hash of provider, model, prompt and parameters, so a repeated run makes no network calls
and returns byte-identical output. The cache is what makes iteration free: re-running
evaluation after a code change costs seconds and nothing.

Throughput is governed by a token bucket rather than a concurrency limit alone. A
semaphore cannot respect a tokens-per-minute ceiling, and dropped calls under rate
limiting are invisible — a run finishes, reports success, and quietly extracted less
than it should. Rate-limit responses are retried with backoff that honours the wait hint
in the response body; schema failures are retried at most twice, because a structural
mismatch does not resolve by asking again.

Each run writes `00_manifest.json` with the git SHA, dataset hash, provider and model
versions, per-stage timings, token counts, cost, and cache hit rate.

### Observability without an answer key

On the private dataset there is no key, so quality has to be visible from the run itself.
Each stage reports counters that indicate extraction health independently of
correctness: unresolved conflicts, unstable fields, clauses that needed a retry, empty
metric legs, the share of breaches, and the distinct kinds of adjustment found. A run
with zero empty legs and a breach share near the expected band is a good extraction; one
with eight empty legs is not, regardless of what the submission looks like.

---

## What the case is actually testing

The dataset is constructed so that computing directly from the ledger produces confident
wrong answers for most borrowers. Reading it closely, three things stand out.

**The answer format is a regulator exam.** Verdict, figure, transaction — that is
exactly what an examiner asks for when they pick a borrower at random and ask the bank
to walk the last covenant cycle: what was decided, what number supports it, which
transaction proves it. A system that produces the verdict but cannot produce the
transaction has not solved the problem the format describes.

**The traps are not puzzles; they are the real failure modes.** Superseded contract
editions that differ by period rather than by threshold. Auditor reclassifications that
move an amount between line items and so change both sides of a ratio. Obligations
disclosed in prose and never posted. A working paper that reads authoritatively and
declares in its own text that it carries no verified figures. Each of these is a way
covenant monitoring actually goes wrong in a bank, compressed into a synthetic archive.

**Clause numbering carries no meaning.** The same paragraph number is a different
covenant for each borrower — interest coverage for one, capital intensity for another,
a springing leverage test for a third. Nothing can be keyed on the slot; each contract
has to be read. The same applies to thresholds, which are stated per borrower, including
the ownership percentage above which a counterparty becomes a related party. Near-miss
entities sit deliberately just below that line, and in one case the entity below the
threshold transacts ten times more than the one above it, so a single misclassification
does not degrade the answer — it inverts it.

Two details reward reading the documents rather than the data. Related-party payments
appear as ordinary consulting fees; the only signal is counterparty identity from the
compliance dossier, and no amount of description parsing recovers it. And the verdict is
decided on the unrounded value while the reported figure is rounded — two cells share a
threshold and a rounded actual yet have opposite verdicts, which is only reproducible if
the comparison happens before the rounding.

The evidence field encodes a further distinction worth stating: the transaction that
proves a breach is the one whose reclassification, inclusion or exclusion changes the
verdict — not the largest contributor, and not the row that happened to carry a running
total past the limit. That is a counterfactual test, and it is implemented as one:
remove each candidate row, recompute, and keep the row whose removal flips the answer.

---

## Results

Measured on the public dataset against the provided answer key.

| | |
|---|---|
| Score | **34.94 / 36.00** |
| Status accuracy | 35 / 36 |
| Ceiling | 35.00 — one cell is unreachable, its source contract states a threshold that contradicts the key |
| Runtime, cold cache | ~15 min (ingest 105 s, classification 63 s, remainder extraction) |
| Cost per run | ~$0.15 on `gpt-4o-mini` |

Extraction is not fully deterministic across cold runs. The final submission is selected
from several extractions by the quality counters described above, rather than by a
single pass.

---

## Requirements

Python 3.11 or later, an API key for either provider, no GPU.

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "covenant-agent"
version = "0.1.0"
description = "Bank covenant checking agent"
requires-python = ">=3.11"
dependencies = [
    "pdfplumber",
    "pymupdf",
    "duckdb",
    "pandas",
    "pydantic",
    "instructor",
    "openai",
    "anthropic",
    "python-dotenv",
    "pytest",
]

[project.scripts]
covenant = "agent.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["agent*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Environment variables, in `.env`:

```
LLM_PROVIDER=openai          # or anthropic
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

---

## Project layout

```
agent/
  __main__.py        orchestrator, artifact resumption, deadline guard
  cli.py             run / preflight / score / validate
  config.py          provider, model id, limits, contact fields
  models.py          domain model, Decimal throughout
  stages/            s1_ingest … s7_emit
  parsing/           pdf, numbers, tables, entity-name normalisation, categories
  metrics/           leg construction, derived shapes, comparison
  llm/               cached client, token bucket, vision path, Pydantic schemas
  evidence/          quote verification, counterfactual search
  trace.py           evidence trace and its self-verification
  validate.py        output invariants
eval/
  score.py           scoring function, per-component and per-slot report
  diagnose.py        failure attribution, per-cell leg breakdown
tests/
  robustness         nine mutated datasets, replayed without network
  parsing            money formats, sentinels, springing detection
```

---

## Testing

```bash
pytest tests/ -q
```

The robustness suite builds mutated copies of the dataset — a missing dossier, an extra
scenario, an unknown slot, an unmarked PDF, a blanked amount, a duplicated contract —
and asserts that each completes, fills every template cell, and records the expected
conflict. LLM responses are replayed from recorded fixtures, so the suite runs offline
in seconds and a live call during a test raises rather than silently succeeding.

---

## Compliance

Code in this repository was authored with the assistance of Cursor. Every answer in
`submission.json` is produced by the pipeline in this repository, executed as the single
command shown above against the competition dataset. No answer was obtained from an
interactive agent and none was entered by hand. The run manifest records the git SHA,
dataset hash and model versions, so any submission can be reproduced and checked against
the code that generated it.

---

## Team

**macintosh** · Halyk AI Challenge 2026

Three days were spent reading the archive before writing the pipeline that consumes it.
That order was the main decision: the traps in this dataset are not visible from the
ledger, and a system designed against the CSV alone converges on answers that are
internally consistent and wrong.
