# Grounded claim verification over an arXiv corpus

Decide whether a factual claim is true, where the only admissible authority is a
fixed corpus of 50 arXiv abstracts.

The hard part is not answering. An LLM answers instantly, from training data rather
than from the corpus, and produces a fluent, confident, well-cited-looking verdict
whose citations do not exist — a failure indistinguishable from a correct answer at
a glance, and most convincing exactly where the corpus is silent.

**Groundedness here is enforced by code, not by prompting.** Every quoted span the
model returns is mechanically checked to be literally present in the document it
cites. Anything that fails is discarded, and a claim with no surviving evidence
returns *unknown*. A prompt regression can make this system less useful. It cannot
make it ungrounded.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...     # only `verify` needs it
```

## Use

Three programs, split at the network boundary and the index boundary.

```bash
fc-fetch --ids corpus/arxiv_ids.txt --out data     # online, resumable, rate-limited
fc-index --docs data --out index                    # offline and pure; no network
fc-verify --index index --docs data                 # online, interactive
```

```
  50 documents · 50 chunks · built 2026-08-03T13:37Z
  config hash b7c866a

claim> RAG is a promising approach for mitigating hallucination in LLMs

  SUPPORTED
  ─────────────────────────────────────────────────────────────────
  [2309.01431v2] entails
    "Retrieval-Augmented Generation (RAG) is a promising approach
     for mitigating the hallucination of large language models"
    Benchmarking Large Language Models in Retrieval-Augmented Generation · 2023-09-04

  retrieved 8 · quotes 1 · span-rejected 0 · 2 LLM calls · 1.8s

claim> :why                    # retrieved set, fusion ranks, per-quote audit stances,
claim> :set rerank.top_n 12    # and why each rejected span failed
claim> :set chunk.size 256     # refused — index-time, names the rebuild command
```

Also `--claim` (one-shot), `--claims-file` (batch), and `--json`.

## How a verdict is reached

```
claim
  ↓  QueryExpander        llm · includes a negation form of the claim
  ↓  BM25 + Dense         lexical for exact tokens, embeddings for paraphrase
  ↓  RRF fusion           GATE · consumes ranks only
  ↓  Reranker             precision stage
  ↓  Verifier             llm · sees only retrieved passages; tool-use schema
  ↓  SpanValidator        GATE · every quote must be verbatim in the doc it cites
  ↓                             zero survivors → unknown, short-circuit
  ↓  EntailmentAudit      llm · one quote + the claim, in isolation
  ↓  Aggregator           GATE · fixed rule over stances
VerificationResult
```

The two gates are deterministic code. Neither is a prompt, and neither is
swappable — a swappable guarantee is not a guarantee.

| Stances present | Verdict | Flag |
| --- | --- | --- |
| contradicts | contradicted | — |
| entails | supported | — |
| entails + contradicts | contradicted | `conflicting_evidence` |
| neither / empty | unknown | — |

**Absence is not contradiction.** Collapsing those last two rows is the standard
naive failure.

## Why the negation query is a correctness decision

Retrieval optimizes for *similarity*, but fact-checking needs *refutation*. Semantic
search returns passages that look like the claim — which is precisely the evidence
that confirms it. Without a negation-form query, refuting passages are rarely
retrieved and the verifier is **structurally incapable** of returning *contradicted*,
regardless of how good it is. The metric that exposes this is per-class recall, not
accuracy.

## Configuration

One frozen tree, hashed, recorded on every evaluation row. Index-time fields are a
different *type* from query-time fields, so varying one against a prebuilt index is
an error rather than a plausible number.

There is deliberately no `validator.fuzzy_match_threshold`. Normalization knobs
adjust *how* text is compared; none adjusts *whether* the quote must be present.
Config tunes quality, never the invariant.

`config/default.yaml` is the profile tuned for **this** corpus. The code defaults are
deliberately scale-neutral: `bm25.min_score` defaults to `0.0` in code because BM25
scores move with corpus size and document length, so a floor baked into the code is a
per-corpus constant pretending to be a default — and its failure mode on an untuned
corpus is an empty result set, which reads as appropriate abstention.

## Evaluation

```bash
fc-eval --index index --docs data --gold eval/gold.jsonl --split dev
```

Retrieval recall is reported **separately** from verdict accuracy, because a wrong
verdict has two unrelated causes — the retriever never surfaced the document, or the
verifier misread it — and you cannot fix what you cannot attribute. That is why
`retrieved` is on the public result: it looks like a debugging leak and is the most
important field for evaluation.

Metrics are per-class, with bootstrap confidence intervals, on a dev/test split. The
harness reports; it does not assert thresholds. A ~30-claim set cannot resolve a
three-point difference — that is roughly one claim flipping — and a build that fails
on statistical noise trains people to ignore the build.

## Honest limitations

- **The default embedder is lexical, not semantic.** `HashingEmbedder` is a hashed
  bag-of-features model chosen so nothing requires a running service or a 2 GB
  download. It catches morphological and vocabulary-overlap paraphrase; it does *not*
  catch true synonym-level paraphrase over disjoint vocabulary. Its on-topic/off-topic
  cosine margin is thin (~0.63 vs ~0.52), which is why the floor needed measuring
  rather than guessing. `indexing/embed.py:get_embedder` is the one-line swap point
  for a trained encoder, and nothing above the retrieval interface changes.
- **The labeled set is the real bottleneck**, not the search strategy. A tunable
  config plus a small labeled set is an efficient machine for overfitting, which is
  why the CIs and the held-out split are specified rather than optional.
- **Scale is designed for, not demonstrated.** Every row of the swap table below is
  an implementation change behind an unchanged interface, but none is implemented.

| Stage | Today (50 abstracts) | At 10⁸ | Interface change |
| --- | --- | --- | --- |
| Lexical | `rank_bm25`, in-process | OpenSearch / Elasticsearch | none |
| Dense | numpy matmul | HNSW / IVF-PQ — pgvector, Qdrant | none |
| Fusion | RRF over ranks | RRF over ranks | none |
| Rerank | pass-through | cross-encoder | none |
| Chunking | chunk == abstract | windowed splits | none — offsets carried from day one |

## Tests

```bash
.venv/bin/pytest -q          # hermetic: no network, no API key, scripted model
.venv/bin/mypy && .venv/bin/ruff check src tests
```

Tests drive the single public seam, `FactChecker.check`, and assert on the returned
result. They do not assert on which stage produced what — internal boundaries are
expected to change as implementations are swapped, and a test that freezes them
converts the extensibility this design is for into a liability.

## Corpus note

The 50 requested identifiers resolve to **48 distinct papers**: `2307.03172` and
`2403.01432` each appear at two versions, and both pairs have textually different
abstracts under an identical title and published date. This was verified against the
live API, not assumed, and it is the origin of the `conflicting_evidence` flag — a
claim keyed to a revised figure can legitimately draw opposite stances from two
versions. Deduplicating hides a real disagreement; letting contradiction silently win
misreports a superseded finding. The system reports both sides with version IDs.
