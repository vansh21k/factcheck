# Gold-claim test report

Run via `python eval/run_gold.py` against the real corpus (`index/`, `data/`),
the real `gemini-flash-latest` client, and the real pipeline (`FactChecker.check`,
the same seam the CLI and unit tests use). Not a synthetic run: every result below
is what the deployed system actually returned.

## Status: incomplete, and why

**6 of 19 claims completed. 13 are blocked, not "slow" — retrying does not help.**

Every call to `gemini-3.6-flash` (what `gemini-flash-latest` currently resolves to)
failed after claim 4 with `429 RESOURCE_EXHAUSTED`, `limit: 20, model:
gemini-3.6-flash`. This survived a bounded retry (3 attempts, 20s backoff) and a
45s inter-claim pacing sleep identically, regardless of the API's own suggested
`retry in Ns` delay — which is the signature of a **daily** cap, not a per-minute
throttle that a bit of pacing routes around. One claim costs 3-4 LLM calls
(expand, verify, one audit per surviving quote), so at roughly 20 requests/day,
**at most ~5 claims can complete per day on this key's free tier, full stop** —
no amount of retry/backoff tuning changes that arithmetic. `eval/run_gold.py` now
writes incrementally and skips claims that already have a real result, so
re-running it tomorrow (or after enabling billing on the underlying Google Cloud
project, which removes the free-tier cap) picks up exactly where this left off
rather than re-paying for the 6 already-answered claims.

| | |
|---|---|
| Completed | 6 / 19 |
| Correct | 3 |
| Wrong verdict | 3 |
| Blocked by quota (not run) | 13 |

## The headline finding

**All 3 mismatches are `contradicted → unknown`. Both correctly-resolved claims
that had anything to retrieve were `supported`/`unknown` and matched.** That's not
noise from a 6-row sample — it's the exact failure mode `HANDOFF.md` predicted
before any of this ran: *"Likeliest first failure is under-retrieval of refuting
evidence — high overall accuracy, poor recall specifically on `contradicted`."*
The data now backs that prediction, and attributes it more specifically than the
original guess:

| claim_id | expected | actual | doc retrieved? | quotes proposed | quotes accepted |
|---|---|---|---|---|---|
| short-01 | supported | supported ✓ | yes | 1 | 1 |
| short-02 | contradicted | unknown ✗ | yes (2004.04906v3) | 1 | **0** |
| long-02 | contradicted | unknown ✗ | yes (2109.07958v2) | 2 | **0** |
| contra-01 | contradicted | unknown ✗ | yes (2109.07958v2) | 1 | **0** |
| out-01 | unknown | unknown ✓ | n/a (0 retrieved) | 0 | 0 |
| out-02 | unknown | unknown ✓ | n/a (0 retrieved) | 0 | 0 |

Retrieval is not the failure. The README's own design principle —
*"`retrieved` is on the public result... a wrong verdict has two unrelated causes,
and you cannot fix what you cannot attribute"* — is exactly the tool that
localizes this: **the correct document was retrieved in all three mismatches.**
The verifier saw the right passage every time. What happened next is the
verifier *proposed* a citation in all three cases (never zero evidence) and the
deterministic `SpanValidator` **rejected 100% of them** — every proposed quote
failed the verbatim-substring check and was discarded, which is what triggers
the `no surviving evidence → unknown` short-circuit.

This is the grounding guarantee working exactly as designed: an unverifiable
citation is discarded rather than trusted. But the net effect, on this small
sample, is that `gemini-flash-latest` as the verifier model appears to paraphrase
rather than copy verbatim specifically when constructing a *refuting* quote —
0/3 accepted vs. 1/1 accepted on the one completed `supported` claim. Three
claims is not enough to call this a proven model-specific weakness rather than
chance; it is enough to say the mechanism is now visible in the stats rather
than hypothetical, and that the next ~13 claims (once quota allows) will either
confirm or break the pattern. `long-02` in particular proposed *two* quotes and
both were rejected, which weakly favors "the model doesn't quote precisely
enough for this kind of claim" over "got unlucky once."

## A second, independent finding: two real gaps in the existing code

Investigating *why* `out-01`/`out-02` show `llm_calls: 0` despite having a
negation-aware expander enabled turned up two things worth fixing, neither
introduced by this session's Gemini work:

1. **`llm_calls` undercounts by one on every claim.** `FactChecker.from_config`
   wraps only `verifier_llm` and `auditor_llm` in `CountingLLM`
   (`checker.py:90-92`); the expander's LLM call happens inside
   `RetrievalPipeline`/`NegationAwareExpander`, using an `expander_llm` that was
   never passed through a counter. Every claim that goes through the default
   `NegationAwareExpander` makes one real, billed API call that the `llm_calls`
   stat (shown on every CLI result line, e.g. `README.md`'s own `2 LLM calls`
   example) never counts.
2. **`NegationAwareExpander.expand()` swallows all exceptions, including
   `ModelCallError`** (`retrieval/expand.py:100`, a bare `except Exception:`),
   silently degrading to a claim-only query with no negation form and no signal
   to the caller that anything went wrong. The docstring is explicit that the
   negation query is "a correctness requirement, not a recall optimization" and
   that without it "the verifier is... structurally incapable of ever returning
   `contradicted`" — which means a transient quota or network failure on the
   expander call doesn't just degrade quality quietly, it can specifically and
   silently disable the one query path `contradicted` depends on, with no flag
   or warning surfaced anywhere in `VerificationResult`. Combined with finding
   (1), this failure mode is currently invisible: the call that can silently
   fail is also the one call nothing counts.

Neither is fixed here — flagged for a decision, not patched blind, since (2) in
particular is a design choice (fail open vs. fail loud) rather than an obvious bug.

## What's still blocked

`long-01`, `long-03`, `negation-01`, `paraphrase-01`, `adversarial-01`,
`misattribution-01`, `quant-01`, `quant-02`, `multidoc-01`, `version-01`,
`subjective-01`, `selfrag-01`, `selfrag-02` — designed, not yet run. Several of
these (`negation-01` especially, and both `quant-*`) bear directly on the
headline finding above and are the next evidence to collect once quota resets.
`adversarial-01` (the prompt-injection claim) is unrun and untested — no claim
is made here about grounding holding under injection until it actually has.

## Spot checks (edge cases outside the gold file)

- **Empty-string claim**: does not reach the pipeline at all.
  `factcheck.evaluation.dataset.load_gold` rejects it at parse time
  (`'claim' must be a non-empty string`) if put in a gold row, and
  `fc-verify --claims-file`'s batch mode skips blank lines before dispatch
  (`cli/verify.py:_batch`). Confirmed directly against `load_gold`, no LLM call
  needed. **Live pipeline behavior for an empty/whitespace-only claim passed
  directly to `FactChecker.check` (bypassing both guards) is untested** — it
  would cost an expander call and is blocked by the same quota wall as
  everything else above.
- Single-word / gibberish / very-long-input spot checks: **not run**, same
  reason. `long-03` in the gold set already stresses input length once quota
  allows it to run.

## Next steps, in order

1. Either enable billing on the Google Cloud project behind the current key
   (removes the free-tier cap) or wait for the daily reset and re-run
   `python eval/run_gold.py` — it resumes for free from the 6 already-cached
   claims.
2. Once `negation-01` and the `quant-*` pair complete, revisit the headline
   finding: does the 0/3 accepted-quote pattern on `contradicted` claims hold,
   or was 3 claims not enough to see it break?
3. Decide on the two code gaps above (wire the expander into `CountingLLM`,
   decide fail-open-silently vs. fail-loud-with-a-flag on expander errors) —
   not done in this pass because they're judgment calls, not obvious bugs.
