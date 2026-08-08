# Covenant Agent — Architecture

An agent that checks corporate loan covenants against an archive of documents and a
transaction ledger. Outputs `submission.json` in the template's shape plus `trace.json`
for debugging.

Figures in this document are measured on the public dataset, not estimated.

---

## 0. Principles

1. **The model extracts; the code decides.** The model returns verbatim spans.
   Arithmetic, comparison and aggregation happen in Python over `Decimal`.
2. **No figure without a source.** Document, page, exact quote. A quote that is not a
   substring of its page is a hallucination and is rejected.
3. **Status is decided on the unrounded value.** Rounding happens only on output. Two
   cells in the key share a threshold and a rounded actual yet have opposite verdicts.
4. **Thresholds are never hardcoded.** Every threshold, including the ownership
   percentage that defines a related party, is read from its own document.
5. **Reading modality is detected per page.** Text layer, image page inside a textual
   document, fully scanned file.
6. **Nothing aborts a stage.** Failures are contained at the level of one clause or one
   document and recorded as conflicts.
7. **No cell is ever empty.** An empty answer scores zero; a guessed one may not. The
   degradation ladder always produces a value.
8. **Nothing is discarded silently.** Unrecognised input goes to a list, never to
   `/dev/null`.

---

## 1. Domain model

```python
@dataclass(frozen=True)
class Provenance:
    doc_id: str
    page: int
    quote: str
    source_kind: str            # "text" | "ocr"
    extractor: str

@dataclass(frozen=True)
class CategorySpec:
    """What belongs to a line item. Defined by the covenant, not by a global taxonomy."""
    include_keywords: list[str] # drawn from the ledger's own categories
    exclude_keywords: list[str]
    sign: str                   # OUTFLOW | INFLOW — derived in code, not asked of the model
    apply_reclass: bool = True

@dataclass(frozen=True)
class MetricSpec:
    kind: str                   # RATIO | SUM
    numerator: "LegSpec"
    denominator: "LegSpec | None"
    scope: str                  # BORROWER | GROUP
    notes: str

@dataclass(frozen=True)
class LegSpec:
    """A leg is either a ledger filter or a derived shape the engine builds itself."""
    shape: str                  # CATEGORY | EBITDA | ADJUSTED_EBITDA | RELATED_PARTY
    category: CategorySpec | None

@dataclass(frozen=True)
class SpringingCondition:
    metric: MetricSpec
    operator: str
    value: Decimal
    source: Provenance

@dataclass(frozen=True)
class Covenant:
    scenario_id: str
    slot: str                   # "6.1" | "6.2" | "6.3" | whatever the template carries
    title: str
    direction: str              # MAX (≤ threshold) | MIN (≥ threshold)
    threshold: Decimal
    threshold_unit: str         # USD | RATIO
    metric: MetricSpec
    period: tuple[date, date]   # read from the contract, never assumed
    springing: SpringingCondition | None
    source: Provenance

@dataclass(frozen=True)
class ClassificationRule:
    """The rule printed beneath a dossier table. Its semantics differ per borrower."""
    semantics: str              # RELATED_PARTY | UNRESTRICTED_SUBSIDIARY
    threshold_pct: Decimal | None
    direction: str              # AT_OR_ABOVE | BELOW
    source: Provenance

@dataclass(frozen=True)
class Adjustment:
    kind: str                   # see section 4
    scenario_id: str
    txn_id: str | None
    amount: Decimal | None
    counterparty: str | None
    from_category: str | None
    to_category: str | None
    target_leg: str | None      # an adjustment applies to a named leg, not to a covenant
    floor: Decimal | None       # EBITDA_ADDBACK
    rows: list[dict] | None     # EBITDA_ADDBACK
    rate: Decimal | None        # FX
    match_method: str
    source: Provenance
```

Two design points carry most of the weight.

**A leg is a shape, not only a filter.** Several covenants divide by EBITDA, which is
not a ledger category and cannot be selected from one. Making derived shapes
first-class removes an entire class of failure where the model, asked to pick
categories, either invents `EBITDA` as a keyword or falls back to plain operating
expenses.

**An adjustment names the leg it targets.** An EBITDA add-back belongs in the numerator
and nowhere else. Applying an adjustment at covenant level rather than leg level silently
double-counts it.

---

## 2. Pipeline

```
data/{open|private}/
├─ 00_manifest.json      run: git sha, provider, models, input hash, timings, cost
├─ 01_inventory.json     text per page, ocr_pages, unreadable list
├─ 02_classified.json    document type, ACC ids
├─ 03_bound.json         scenario → its documents
├─ 04a_covenants.json    one covenant per template cell
├─ 04b_parties.json      classification tables and their rules
├─ 04c_adjustments.json  adjustments and the unrecognised list
├─ 05_ledger.parquet     filtered, adjusted, converted
├─ 06_evaluated.json     verdicts, values, evidence, review list
├─ trace.json            evidence trace
└─ submission.json       projection of the trace
```

Each stage reads one artifact and writes the next. A stage skips when **all** of its
outputs exist, so deleting one artifact forces exactly that stage to run. Checking a
single marker file is a subtle bug: deleting `04a_covenants.json` then appears to
succeed while the stale artifact is silently reused.

---

## 3. Stages

### 1 — Ingest

Walk the input directory. For each PDF: `sha256`, page count, text per page via
`pdfplumber`. Non-PDF files inside the documents folder are read into the same
inventory — in the public dataset a `.txt` file carried an operative rule about
superseded editions. Unreadable types go to an `unreadable` list and never raise.

**Image detection is per page.** A page whose stripped text is under 100 characters
while carrying an image is rendered at 150 DPI and recorded in `ocr_pages` with its page
number. Gating on the first page finds fully scanned files and misses image pages inside
otherwise textual documents — seven such pages across four documents in the public set,
each carrying load-bearing content.

### 2 — Classify

Filenames are opaque hashes. Classification is by content marker, first match wins, in
this order:

```
SUPERSEDED_DRAFT   superseded by final | draft | not to be relied upon
LOAN_SUPERSEDED    superseded | prior version | no longer in force
LOAN               execution copy | senior secured loan
AUDIT_NOTES        notes to the financial statements | audit file
KYC                know your customer | customer due diligence
ADJUSTMENT_SOURCE  agreed-upon procedures | treasury memorandum
AUDIT_PLANNING     external audit — planning memorandum
NOISE              anything else
```

Every marker exists in both Russian and English, matched case-insensitively. The
organisers confirmed the private dataset mixes both languages; a single-language marker
set drops an entire borrower when its contract arrives in the other one.

**Order encodes two traps.** `SUPERSEDED_DRAFT` is tested first because working papers
carry plausible numbered adjustments and are all superseded, while a genuine treasury
memorandum is also labelled a working document — the disclaimer block is the
discriminator, not the label. `AUDIT_PLANNING` is classified explicitly and then
excluded from fact extraction: it reads authoritatively and states in its own text that
it contains no verified figures.

A document whose text layer matches no marker but which has rendered pages is classified
by sending its first rendered page to the vision model with the same marker list. Without
this a scanned dossier lands in `NOISE` and its borrower loses the binding.

### 3 — Bind

Build `scenario_id → {loan, audit_notes, kyc, adjustment_sources}`.

The account is derived from the ledger: the prefix of `txn_id` is the scenario id and
`account_id` sits in the same row.

```python
acc2scen = (ledger.assign(sc=ledger.txn_id.str.split("-").str[1])
                  .groupby("account_id").sc.agg(lambda s: s.mode()[0]).to_dict())
```

**Never bind by company name.** The public dataset contains
*Shymkent Refinery Services JSC* alongside *Shymkent Refinery JSC*, and
*Ekibastuz Power Services JSC* alongside *Ekibastuz Energy JSC*.

A dossier belongs to the account named in its header block, not to every account id
appearing in its body — other documents reference accounts in passing.

Invariant: exactly one operative loan per scenario.

### 4a — Covenants

Cut Article 6 from the operative contract, split by clause marker, extract each clause
into a `Covenant`.

**The section heading appears twice** — once in the table of contents and once as the
real heading. Matching the first occurrence extracts a fragment of the contents page and
fails silently. Take the last occurrence and assert the section contains every expected
clause marker.

`direction` is an explicit field, never inferred from the title. The same slot appears
as a minimum revenue floor for one borrower and a maximum category ceiling for another;
getting the direction wrong zeroes the whole cell.

The covenant period is read from the contract rather than assumed. Contracts state their
own period, and a period that does not match the reporting year usually means a
superseded edition slipped through classification.

Springing conditions are detected by trigger phrase before the model is asked about
them. No trigger phrase means `None` without a call — which removes the failure mode
where the model returns a springing object with blank required fields.

### 4b — Classification tables

A dossier carries whichever table that borrower's covenants need. Two forms appear:
ownership percentages with a relatedness threshold, and pledged-asset percentages with a
security-perimeter threshold below which a subsidiary counts as unrestricted. Extract the
table together with the rule sentence printed beneath it and store the semantics.

**Thresholds are per borrower.** Values observed: 25.0, 35.0 and 40.0 for relatedness,
50.0 for the security perimeter. Comparison is exact and never rounds. Entities sitting
just below a threshold are deliberate decoys, and in one scenario the near-miss entity
transacts an order of magnitude more than the genuine related party — a single
misclassification does not degrade the metric, it inverts it.

Three dossiers keep this table on an image page while the rest of the document is text.
Those pages go to the vision model under the same schema. Quote verification cannot apply
to an image, so those fields are marked `source_kind="ocr"` and listed for review.

Counterparty matching normalises case, whitespace, quotation marks, dotted legal forms
(`L.L.P.` as well as `LLP`) and a comma preceding the suffix. **No fuzzy matching** — the
dataset contains distinct entities with near-identical names.

Related-party payments cannot be identified from transaction descriptions. They appear as
ordinary consulting fees. Counterparty identity is the only discriminator.

### 4c — Adjustments

Sources are `AUDIT_NOTES` and `ADJUSTMENT_SOURCE`. `SUPERSEDED_DRAFT` and
`AUDIT_PLANNING` are excluded.

Two segmentation paths, both required.

*Numbered items.* Markers vary — the number tracks the note they sit under, and treasury
memoranda use a bare index with no dot. Where a covenant-compliance heading exists,
extraction is restricted to the section following it.

*Image pages.* Some adjustments carry no marker at all and live on a rendered page. After
marker segmentation, every `ocr_page` of every source document is read by the vision
model.

### 5 — Ledger

```
filter by the scenario's account_id
    ↓
AMOUNT_FILL      fill empty amounts on existing rows
    ↓
EXCLUDE, CUTOFF  mark rows excluded
    ↓
RECLASS          change a row's category
    ↓
FX               convert at the derived rate
    ↓
OFF_LEDGER       append synthetic rows
```

**Order changes the result.** `AMOUNT_FILL` runs first: a row with an empty amount is
invisible to every aggregate that ran before it, and pandas drops it silently rather than
raising. Invariant: no row retains a null amount after adjustments.

There is no global category taxonomy. Membership is decided when a covenant is evaluated,
from that covenant's own definition. The same row counts toward different line items for
different borrowers.

### 6 — Evaluate

```python
def evaluate(cov, ledger, entities, adjustments) -> Finding:
    if cov.springing:
        trigger = compute(cov.springing.metric, ledger, entities)
        if not compare(trigger, cov.springing.operator, cov.springing.value):
            return Finding(status="COMPLIANT",
                           evaluated=compute(cov.metric, ledger, entities),
                           evidence=None, strategy="springing_not_triggered")

    actual = compute(cov.metric, ledger, entities)          # Decimal, full precision
    status = "BREACH" if breaches(actual, cov.direction, cov.threshold) else "COMPLIANT"
    evidence = find_evidence(cov, ledger, status) if status == "BREACH" else None
    return Finding(status, evaluated=actual,
                   rounded=round_half_up(abs(actual), 2), evidence=evidence)
```

**Legs use signed arithmetic.** EBITDA is revenue minus operating expenses. Taking the
absolute value of each row before summing turns every difference into a total; only the
final reported figure is made positive.

**Comparison precedes rounding.** Rounding first collapses boundary cases that the key
distinguishes.

**A springing covenant whose trigger is not met still reports its real metric value**,
which may legitimately exceed the limit. Not `null`, and not the trigger value.

**An empty or zero leg is never divided by.** Under a MAX covenant a zero result always
reads as compliant, which converts a missing value into a confident wrong answer. Such
cells route through the degradation ladder instead.

**Both legs of a ratio drawing the same rows is an extraction failure**, not a value.
A share has a whole and a part; if they resolve identically, something was selected
twice.

#### Evidence is a counterfactual

The evidence transaction is the one whose reclassification, inclusion, exclusion or
correction changes the verdict. Not the largest contributor, not the last before period
end, and not the row that carried a running total past the limit.

```python
def find_evidence(cov, ledger, status):
    candidates = ledger[relevant_to(cov)]        # scoped to this scenario first
    flips = [t.txn_id for t in candidates.itertuples()
             if verdict(compute(cov.metric, ledger.drop(t.Index)), cov) != status]
    if len(flips) == 1:
        return flips[0]
    if len(flips) > 1:
        adjusted = [t for t in flips if t in adjusted_txn_ids]
        return adjusted[0] if len(adjusted) == 1 else None
    return None
```

Preferring a row touched by an auditor adjustment is not a heuristic: in the public
dataset the evidence for one cell is precisely the amount the auditor reclassified.

Invariant: non-null evidence implies BREACH. The converse does not hold.

#### Degradation ladder

| Field | Rungs |
|---|---|
| `status` | computed → slot prior |
| `actual` | computed → the threshold value → 0 |
| `evidence` | counterfactual → `null` |

The rung that fired is recorded in `strategy`. Anything other than `computed` appears in
the review list.

### 7 — Emit and validate

`submission.json = project(trace)` — a pure function, no model calls, no PDF reads. The
template is filled in place; keys are never renamed, added or removed.

Validation records conflicts and never aborts. An imperfect submission scores; a missing
one does not. The count of failed verifications is reported so quality stays visible
without blocking output.

Checks: keys identical to the template, `status` exactly one of the two literals,
`actual` positive with two decimals, `evidence_txn_id` present in the ledger or null,
non-null evidence implies BREACH, every threshold carrying a verified quote, no ledger
row with a null amount, the unrecognised-adjustment list present in the output.

---

## 4. Adjustment taxonomy

| Kind | Effect on the ledger |
|---|---|
| `RECLASS` | moves a row between categories |
| `CUTOFF` | excludes a row whose service period falls outside the covenant period |
| `EXCLUDE` | excludes a row from the period |
| `OFF_LEDGER` | appends a synthetic row for an amount disclosed but never posted |
| `AMOUNT_FILL` | supplies the figure for an existing row whose amount column is empty |
| `FX` | provides a rate derived from a disclosed settlement |
| `EBITDA_ADDBACK` | one-off items above a materiality floor return to the numerator |
| `NONE` | an item explicitly stating no adjustment was required |

`OFF_LEDGER` and `AMOUNT_FILL` differ in the operation, not the amount. The first creates
a row that does not exist; the second edits one that does and keeps its real transaction
id, which makes it eligible as evidence.

`EBITDA_ADDBACK` carries rows and a materiality floor. The one-off items are expenses:
they are subtracted inside operating expenses first, then only those at or above the
floor are added back. Adding back an amount that was never subtracted inflates EBITDA —
and the presence of an add-back is itself proof the items sat in expenses.

`FX` has no rate table. The source discloses a settlement — an invoice in one currency
paid in another — and the rate is derived by division and stored at full precision.

`NONE` items are recorded and never applied. Anything the classifier cannot map goes to
the unrecognised list: a dataset from the same generator may carry a kind not seen here,
and a silent drop is how it goes unnoticed.

---

## 5. Failure containment

Every stage contains failures at the smallest meaningful unit.

One clause failing to extract records `EXTRACTION_FAILED` with its scenario, slot and
message, and the remaining clauses continue. One dossier or one adjustment document
failing does not stop the others. A verification mismatch in stage 7 records a conflict
and still writes the submission.

The reasoning is arithmetic: one covenant is worth a fraction of a point; a stage is
worth everything. A run that reports thirty-five good answers and one flagged failure is
strictly better than one that reports nothing.

The same principle governs schema design. A model that answers `<UNKNOWN>` for a figure
it cannot find is behaving correctly — sentinel values are recorded as absence, not
raised as parse errors, and retrying them never changes the answer. Losing one field must
never discard the whole record.

---

## 6. Determinism, throughput and cost

`temperature=0`, fixed seed, pinned model id. Every request is cached on disk under a
hash of provider, model, prompt and parameters. A repeated run makes no network calls and
returns byte-identical output — which is what makes iteration free: re-running evaluation
after a code change costs seconds and nothing.

Throughput is governed by a **token bucket**, not a concurrency limit alone. A semaphore
cannot respect a tokens-per-minute ceiling, and dropped calls under rate limiting are
invisible: the run finishes, reports success, and quietly extracted less than it should.
Requests are estimated before sending and blocked until budget allows.

Retry policy distinguishes error classes. Transport failures retry with exponential
backoff honouring the wait hint in the response body. Schema failures retry at most
twice, because a structural mismatch does not resolve by asking again — and each retry
appends the previous attempt to the prompt, so later attempts answer a different question
than the first.

Provider is one environment variable. Both OpenAI and Anthropic paths are exercised,
including the vision path, whose image payload format differs between them and is hidden
behind a single helper.

Each run writes `00_manifest.json` with git SHA, dataset hash, provider, model and parser
versions, per-stage timings, token counts, cost and cache hit rate.

---

## 7. Observability without an answer key

On unseen data there is no key, so extraction quality has to be readable from the run
itself. Each stage reports counters that indicate health independently of correctness:

```
s4a_covenants:  extracted  springing  conflicts  unstable_fields  retry_clauses  extraction_failed
s4c_adjustments: adjustments  unrecognised  conflicts  extraction_failed
s6_evaluate:    cells  breaches  empty_legs
s7_emit:        cells  verification_failed
```

`empty_legs` is the strongest single signal: a leg matching no rows means a category was
selected that does not exist in this borrower's ledger. `unstable_fields` and
`retry_clauses` indicate how hard the model had to work to produce an answer. The share
of breaches indicates whether the verdicts are plausible at all — a run at zero or above
eighty percent has broken somewhere.

This matters because extraction is not fully deterministic across cold runs, and the
spread between a good and a poor extraction is roughly an order of magnitude larger than
the effect of any single engine change. The practical consequence is that the final
submission is selected from several extractions by these counters, and that improving the
engine and improving the extraction are separate activities that must be measured
separately — holding the extraction artifact fixed while changing code, and vice versa.

---

## 8. Scoring and diagnostics

`eval/score.py` implements the competition's scoring function and reports per component
and per slot. A single aggregate hides which part is broken: a status error zeroes a cell
regardless of the other fields, so status accuracy and value accuracy must be read
separately.

`eval/diagnose.py` attributes failures rather than reporting them. For each cell it names
the likely cause — a delta equal to a known adjustment amount means that adjustment was
not applied; a ratio near a thousand means a scale error; a related-party leg listing an
entity below the dossier threshold means a decoy was admitted.

`diagnose --cell SCENARIO/SLOT` prints every contributing row with the reason it was
included, the derived terms for legs that are not row filters, the substituted expression
and the comparison. A derived leg prints its terms rather than an empty row list —
printing nothing for a computed leg makes zero and derived indistinguishable, which is the
worst possible property for a debugging tool.

---

## 9. Deliberately not built

**A retrieval layer.** Two hundred documents is not a retrieval problem. Marker matching
is exact and auditable; similarity is the wrong tool where a superseded contract differs
from its replacement by one header line, and it would introduce run-to-run variation in
the one decision — which document a borrower is bound to — where variation is most
expensive.

**A general OCR engine.** The content on image pages is tabular, and layout carries
meaning that character recognition discards. A vision model under the same schema as the
text path costs less integration and reads tables better.

**Multi-agent orchestration.** Seven functions and a loop. Additional agents add latency
and failure modes without adding accuracy on a task whose decisions are all deterministic.

**Fine-tuning.** No labelled pairs exist, the private data differs, and the bottleneck was
never model quality.

**A global category taxonomy.** Categories are defined by each covenant's own text.

**Rounding before comparison.** It collapses exactly the boundary cases the key
distinguishes.

---

## 10. Repository layout

```
agent/
  __main__.py        orchestrator, artifact resumption, deadline guard
  cli.py             run / preflight / score / validate
  config.py          provider, model id, limits, contact fields
  models.py          domain model, Decimal throughout
  stages/            s1_ingest … s7_emit
  parsing/           pdf, numbers, tables, entity names, categories
  metrics/           leg construction, derived shapes, comparison
  llm/               cached client, token bucket, vision path, schemas
  evidence/          quote verification, counterfactual search
  trace.py           evidence trace and its self-verification
  validate.py        output invariants
eval/
  score.py           scoring function, per-component and per-slot report
  diagnose.py        failure attribution, per-cell leg breakdown
tests/
  test_robustness.py       nine mutated datasets, replayed offline
  test_parsing_regression.py money formats, sentinels, springing detection
  test_metrics_engine.py     signed arithmetic, add-back placement, identical legs
```

The frontend, if any, lives outside `agent/`. A defect in a viewer must not be able to
affect the submission.

---

## 11. Reference values

Measured on the public dataset. A stage that does not reproduce these has regressed.

| Check | Expected |
|---|---|
| Document types | 138 noise, 12 loans, 12 superseded loans, 12 audit files, 12 dossiers, 8 planning memos, 5 superseded drafts, 2 adjustment sources |
| Image pages | 7 pages across 4 documents |
| Ledger | 1473 rows, 673 belonging to scenarios, 2 with an empty amount |
| Dossier thresholds | 25.0, 35.0, 40.0 relatedness; 50.0 perimeter |
| `OFF_LEDGER` | 918,447.52 |
| `AMOUNT_FILL` | 486,204.19 and 884,204.16 |
| `EBITDA_ADDBACK` | above-floor sum 824,152.91, one row discarded below the floor |
| `FX` | rate derived by division, 1.16 |
| `RECLASS` | 1,104,663.28 and 592,296.10 |
| Key invariants | 17 breaches of 36 cells; 9 evidence ids, all on breaches; slot 6.2 never carries evidence |

Target: every status correct. Status is a gate — a wrong verdict zeroes the cell whatever
the other fields contain.
