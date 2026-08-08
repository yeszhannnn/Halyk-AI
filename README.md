# Halyk AI covenant agent

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Deterministic](https://img.shields.io/badge/inference-temperature%200-informational)
![Halyk AI Challenge](https://img.shields.io/badge/Halyk%20AI%20Challenge-2026-lightgrey)

An autonomous agent that reads an archive of corporate loan documents together with a
transaction ledger and decides, for every covenant of every borrower, whether it is met.

* **Deterministic arithmetic.** The language model only extracts verbatim spans from
  documents. Every comparison, aggregation and ratio is computed in Python over
  `Decimal`, never by the model.
* **Evidence for every figure.** Each number carries the document, page and exact quote
  it came from, and quotes are machine-verified as substrings of the page they claim.
* **Three reading modalities.** Text layers, image-rendered pages inside otherwise
  textual documents, and fully scanned files are each detected per page and routed
  to the right extractor.

---

## Quick start

```bash
git clone <repo-url> && cd covenant-agent
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .

cp .env.example .env                                  # add OPENAI_API_KEY
```

Place the dataset so that the input directory contains `documents/`,
`master_ledger_2025.csv` and `submission_template.json`, then run:

```bash
python -m agent --input data/private --out submission.json
```

That single command produces the submission. There is no manual step anywhere in the
pipeline, and no path through the code where an answer can be entered by hand.

Useful flags:

```bash
python -m agent --input data/open --out submission.json --force   # ignore cached artifacts
python eval/score.py submission.json eval/ground_truth.json       # score against a key
```

After any prompt or schema change, re-record the mini-dataset LLM replay fixtures
(offline tests depend on them):

```bash
python scripts/record_mini_llm_fixtures.py
```

---

## How it works

Seven stages, each reading one artifact from disk and writing the next. A stage is
skipped when its output already exists, so a failed run resumes rather than restarts.

| Stage | Does | Writes |
|---|---|---|
| 1 · Ingest | text per page, per-page scan detection, page rendering | `01_inventory.json` |
| 2 · Classify | document type from content markers; filenames are opaque hashes | `02_classified.json` |
| 3 · Bind | borrower ↔ documents, resolved through the ledger, never by company name | `03_bound.json` |
| 4b · Classification tables | ownership and pledged-asset tables with the rule stated beneath each | `04b_parties.json` |
| 4c · Adjustments | eight kinds of auditor adjustment, from text and from image pages | `04c_adjustments.json` |
| 5 · Ledger | filter, apply adjustments in order, convert currency; emits category vocabulary | `05_ledger.parquet`, `05_ledger.json` |
| 4a · Covenants | thresholds, comparison direction, metric definitions, springing conditions (category enum from stage 5) | `04a_covenants.json` |
| 6 · Evaluate | applicability gate, full-precision comparison, counterfactual evidence | `06_evaluated.json` |
| 7 · Emit | evidence trace, then a pure projection into the submission template | `trace.json`, `submission.json` |

`submission.json` is a pure function of `trace.json`. Changing output formatting is a
one-function edit and a one-second rebuild, not another pipeline run.

### Non-obvious cases handled

The dataset is built so that computing directly from the ledger gives wrong answers.
The agent handles, among others:

* Superseded contract editions, distinguished from the operative one by covenant period.
* Auditor reclassifications that move an amount between line items, changing both the
  numerator and the denominator of a ratio.
* Amounts disclosed only in prose and never posted as a ledger entry.
* Ledger rows that exist with an empty amount column, supplied by a separate document.
* One-off items added back to EBITDA only above a materiality floor stated in the source.
* Foreign-currency rows where no rate table exists and the rate must be derived from a
  disclosed settlement.
* Related-party status resolved from ownership tables whose threshold differs per
  borrower, with entities placed deliberately just below it.
* Working papers that carry plausible figures and are marked superseded.

### What the model is not allowed to do

* It does not perform arithmetic. Every figure is computed in Python.
* It does not decide which contract edition applies. That is a rule over document markers.
* It does not choose the evidence transaction. Evidence is found by removing each
  candidate row and keeping the one whose removal changes the verdict.
* It does not report a value it cannot quote. Extracted spans are verified against the
  source page and rejected on failure.

---

## Determinism

Inference runs at `temperature=0` with a fixed seed and a pinned model id. Every request
is cached on disk under a hash of model, prompt and parameters, so a second run over the
same input makes no network calls and returns identical output.

Each run writes `00_manifest.json` recording the git SHA, dataset hash, model and parser
versions, per-stage timings, token counts and cost.

---

## Results

Measured on the public dataset with the provided answer key.

| | |
|---|---|
| Score | <!-- fill after final run --> |
| Status accuracy | <!-- fill --> |
| Runtime, cold cache | <!-- fill --> |
| Cost per run | <!-- fill --> |

---

## Project layout

```
agent/
  __main__.py        orchestrator, artifact resumption, deadline guard
  config.py          model id, paths, limits, contact fields
  models.py          domain model, Decimal throughout
  stages/            s1_ingest … s7_emit
  parsing/           pdf, numbers, tables, entity-name normalisation
  llm/               cached client, vision path, Pydantic schemas
  evidence/          quote verification, counterfactual search
  validate.py        output invariants
eval/
  score.py           scoring function and per-component report
tests/
```

---

## Requirements

Python 3.11 and an OpenAI API key. No GPU. Dependencies are pinned in `pyproject.toml`;
`pdfplumber` and `PyMuPDF` for documents, `duckdb` and `pandas` for the ledger,
`pydantic` and `instructor` for typed extraction.

---

## Compliance

Every answer in `submission.json` is produced by the pipeline in this repository, executed as the single
command shown above against the competition dataset. No answer was obtained from an
interactive agent, and no answer was entered by hand. The run manifest records the git
SHA, dataset hash and model versions so that any submission can be reproduced and
checked against the code that generated it.

---

## Team

**macintosh** · Halyk AI Challenge 2026
