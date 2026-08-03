# Handoff — claim verification system

Written 2026-08-03. Next session starts cold; this is everything needed to begin implementing.

## State

**Nothing is implemented.** The project directory contains only the spec:

```
/Users/vansh/interviews/factcheck/
  HANDOFF.md                             ← this file
  .scratch/claim-verification/spec.md    ← the spec, Status: ready-for-agent
```

Not a git repo yet. No venv, no dependencies, no source files.

## Read first

1. `.scratch/claim-verification/spec.md` — 90 user stories, implementation decisions, testing decisions, scope boundaries.
2. Design walkthrough with diagrams: https://claude.ai/code/artifact/d5c9d979-080d-4504-add7-46123deb4a5d

The spec is authoritative. This file covers what the spec deliberately leaves out: build order, environment setup, and API corrections discovered after the spec was written.

## Correction to the design — temperature

An earlier version of this design specified **temperature 0 for determinism**. That is wrong for the current models and would fail immediately.

`temperature`, `top_p`, and `top_k` are **removed on Claude Opus 5 and return a 400**. Do not put them in the request. Consequences:

- Test determinism comes from the scripted stub LLM, not from a sampling parameter — which is how the spec's single-seam design works anyway.
- The stage cache key changes from `(stage, inputs_hash, model, temperature)` to `(stage, inputs_hash, model, effort, prompt_version)`.
- Steering that used to be a temperature knob is now `output_config={"effort": ...}` plus prompting.

## Other API facts that shape the code

**Thinking is on by default on Claude Opus 5**, and `max_tokens` caps thinking *plus* response text together. Size `max_tokens` generously (~16000 non-streaming) or the verifier will truncate mid-answer. Omitting the `thinking` field does not disable it.

**Structured output** is `output_config={"format": {"type": "json_schema", "schema": ...}}`. The cleanest path for the verifier's forced-citation schema is `client.messages.parse()` with a Pydantic model — it validates the response against the schema automatically. The deprecated top-level `output_format` parameter is a different thing; don't use it.

**Strict tool use** (`strict: True` on the tool definition, with `additionalProperties: false`) is the alternative if you'd rather express the verifier as a tool call than as a response format. Either satisfies the spec's "citation is a type requirement".

**Handle `stop_reason == "refusal"` before reading `response.content`.** Opus 5 runs safety classifiers; a decline returns HTTP 200 with an empty or partial `content`. Code that indexes `content[0]` unconditionally will break. Cheap to add, annoying to diagnose.

**Prompt caching minimum is 512 tokens on Opus 5.** The verifier's system prompt plus schema likely clears it — worth a `cache_control` breakpoint on the last system block, since the prompt is identical across every claim in an eval sweep.

## Open decision: the audit model

The spec calls for a second-pass entailment audit on a narrow NLI task. I originally wrote "smaller model" into the design; that is your call to make, not an automatic optimization.

- **Default:** `claude-opus-5` for both passes. Correct by default, no capability question.
- **Cheaper:** `claude-haiku-4-5` for the auditor only. The audit sees one quote and one claim in isolation — a genuinely narrow task — and it runs once per surviving quote, so it dominates call volume. This is where the cost lives.

Decide before writing the auditor; it only changes a config value either way.

## Open decision: embeddings provider

Dense retrieval needs vectors, and there is no Anthropic embeddings endpoint to use. Options:

- **`sentence-transformers` locally** — no external service, matches the task's "no infra setup" constraint, works offline. One model download on first run. This is the default unless you have a reason otherwise.
- **A hosted embeddings API** (Voyage, OpenAI) — adds a second key and an external dependency the task explicitly discourages.

Note this is only needed at step 4 below. Steps 1–3 ship a working, grounded app on BM25 alone.

## Build order

Ordered so a functioning, *grounded* app exists as early as possible. The span validator lands in step 3, not later — adding it afterward means demoing an ungrounded system and retrofitting the one guarantee the design is built on.

| # | Step | Delivers |
|---|---|---|
| 1 | `fetch` — arXiv ingest → `documents.jsonl` | Corpus on disk, asserts document count |
| 2 | `index` — BM25 only + manifest | Loadable index, offline/online split proven |
| 3 | `FactChecker` — verifier pass 1 + span validator + aggregator, one-shot CLI | **Functioning grounded app.** Stop here at ~30 min. |
| 4 | Dense retrieval + RRF fusion | Hybrid recall |
| 5 | Query expansion with negation form | Makes `contradicted` reachable |
| 6 | Entailment audit pass 2 | Catches verbatim-but-irrelevant quotes |
| 7 | REPL with `:why` and `:set` | Interactive tuning |
| 8 | Config file + hashing | Sweepable |
| 9 | Gold set + eval harness | Measurable |
| 10 | Grounded autograder + judge calibration | Scales past the labeled set |

Steps 1–3 are the interview deliverable. Everything after is depth to defend.

## Environment

```bash
cd /Users/vansh/interviews/factcheck
git init
python3 -m venv .venv && source .venv/bin/activate
pip install anthropic rank_bm25 pytest
# step 4 only:
pip install sentence-transformers numpy
```

API key: run `ant auth status` first. If it reports an active profile, a bare `Anthropic()` client works with no env var — don't ask for a key. Only if there's no active credential source does `ANTHROPIC_API_KEY` need setting.

## Verified corpus facts

Both confirmed against the live API on 2026-08-03, not assumed:

- **`http://export.arxiv.org` returns 301 with an empty body.** Use `https://` and follow redirects. Without this, ingest silently yields zero documents and every claim returns `unknown` — a failure that reads as appropriate caution rather than a broken corpus. This is why ingest asserts a document count.
- **50 requested IDs resolve to 48 distinct papers.** `2307.03172` and `2403.01432` each appear at two versions. I diffed the `2307.03172` v1/v3 pair: identical title, identical published date, **different abstract text**. This is the origin of the `conflicting_evidence` flag and is invented scope — reasonable to cut, but it's the most interesting thing here to defend out loud.

Working probe:

```bash
curl -sSL "https://export.arxiv.org/api/query?id_list=2309.01431v2,2307.03172v1&max_results=10"
```

## Things not to relitigate

Settled with the user, with reasons, in the spec:

- Python, not Java. The repo's Java rule covers DSA whiteboard code; this isn't that.
- One test seam: `FactChecker.check()`, deps injected.
- Grounding is **both** a deterministic span check and an independent LLM audit — not either/or.
- `SpanValidator` is deliberately not a port and not configurable in its core requirement. There is no fuzzy-match threshold, on purpose.
- The `no evidence → unknown` short-circuit lives above `AggregationPolicy`, so no policy can answer on zero evidence.
- `fetch` / `index` / `verify` are split to separate offline from online concerns.

## Known weak points

- **Thirty gold claims cannot resolve small differences.** A three-point gap is roughly one claim flipping. Config sweeps on a set this size overfit fast — hence dev/test split and bootstrap intervals in the spec.
- **Eleven protocols over 50 documents is more indirection than today's problem needs.** The standing test: a port that never earns an ablation row should be collapsed.
- **Likeliest first failure is under-retrieval of refuting evidence** — high overall accuracy, poor recall specifically on `contradicted`. Query expansion is the designed mitigation and the first thing to measure. This is why metrics are per-class rather than aggregate.
