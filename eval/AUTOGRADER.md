# The autograder: what exists, what doesn't, and what the metrics mean

Short answer first, because the honest one isn't what the name implies:
**there is no autograder in this codebase.** No LLM-as-judge, no scoring model,
no grading rubric. `HANDOFF.md` lists "Grounded autograder + judge calibration"
as step 10 of 10 in the build order, explicitly the last, explicitly the
"depth to defend" tier beyond the interview deliverable — and it was never
built. This doc exists so that gap is a documented TODO instead of a surprise
someone finds by reading source that isn't there.

## What "evaluation" actually means today

Two completely different things share the word "grounding" in this project's
vocabulary, and only one of them is a grader:

1. **`SpanValidator` — real, deterministic, tested code.** Every quote the
   verifier proposes is checked to be a literal, character-for-character
   substring of the document it cites. This is not a judge; it's a string
   search plus the normalization rules in `config/default.yaml`'s
   `validator` block. It runs on every claim, in production, not just in
   eval. It cannot be fooled by a fluent paraphrase, and that's the entire
   point of it existing — see `README.md`'s opening paragraph.
2. **Gold-claim comparison — exact string equality, nothing more.**
   `eval/run_gold.py` runs a claim through `FactChecker.check`, takes the
   returned `Verdict` enum value, and compares it to the hand-written
   `label` in `eval/gold.jsonl` with `==`. That's the entire grading
   mechanism: `actual_verdict == expected_label`. No partial credit, no
   confidence weighting, no second model judging whether the *reasoning* or
   the *cited evidence* was any good — only whether the three-way verdict
   (`supported` / `contradicted` / `unknown`) matches a label one person
   wrote by hand.

Neither of these is what "autograder" means in the original design
vocabulary. The spec's autograder — an LLM that reads a claim, the retrieved
passages, and the system's verdict, and independently judges whether the
verdict was *justified*, at a scale beyond what a hand-labeled set can cover —
does not exist. Nothing in this repo does that.

## What's missing between "exact match" and a real autograder

The gap matters because exact-match-on-label has real, known blind spots this
project's own design otherwise takes seriously:

- **It can't tell "right verdict, fabricated reasoning" from "right verdict,
  solid evidence."** A `supported` that happens to match the label but cites
  a barely-relevant quote scores identically to one with an airtight
  citation. `SpanValidator` guarantees the quote *exists verbatim* — it says
  nothing about whether it's *good* evidence for the specific claim, only
  that the auditor's per-quote entailment stance judged it that way.
- **It can't grade citation quality at all**, only the final aggregated
  verdict. Two systems that agree on `contradicted` for a claim but disagree
  on *which* passage justifies it look identical to this comparison.
- **It doesn't scale past what a human labels.** This is the bottleneck the
  README's original "Evaluation" section already named honestly: *"The
  labeled set is the real bottleneck, not the search strategy."* Nineteen
  hand-written claims is a smoke test, not a benchmark, and no exact-match
  script changes that arithmetic — only a broader, likely LLM-assisted
  labeling or judging process would.

## What the original spec's metrics would have been (never computed)

`README.md`'s `## Evaluation` section documents design intent that predates
this session and was never implemented: `fc-eval` reading `eval/gold.jsonl`,
reporting **per-class accuracy with bootstrap confidence intervals**, on a
**dev/test split**, with **retrieval recall reported separately from verdict
accuracy** — because a wrong verdict has two unrelated causes (the retriever
never surfaced the document, or the verifier misread a document it did see)
and conflating them into one accuracy number destroys the ability to attribute
a regression to a stage. None of that is computed by anything that runs today.
`factcheck.evaluation.dataset.load_gold` (the parser, including the `split`
field on every `GoldClaim`) is real and tested; `factcheck.evaluation.harness`,
the module that would consume it and produce those numbers, doesn't exist —
running `fc-eval` fails with `ModuleNotFoundError` despite being a declared
entry point in `pyproject.toml`.

## What the numbers say today, and how small they are

From the 6 of 19 gold claims that have actually run (`eval/results.json`,
written by `eval/run_gold.py`; see `eval/test_report.md` for the full
write-up including *why* only 6):

| Metric | Value | n |
|---|---|---|
| Overall exact-match accuracy | 50% (3/6) | 6 |
| `supported` recall | 100% (1/1) | 1 |
| `unknown` recall | 100% (2/2) | 2 |
| `contradicted` recall | **0% (0/3)** | 3 |

Do not read more into this table than it can support. Six claims cannot
resolve a difference the way `README.md` already argues thirty can't: *"A
~30-claim set cannot resolve a three-point difference — that is roughly one
claim flipping."* At n=3 for the `contradicted` class, a single claim flipping
is the difference between 0% and 33% recall. What makes the 0/3 worth writing
down anyway isn't the percentage — it's that it lines up with a specific,
inspectable mechanism in the per-claim stats (all three proposed a citation
and had it rejected by `SpanValidator`, none failed to retrieve the right
document), which is a different and more falsifiable claim than "the number
was low." See `eval/test_report.md` for that evidence. Retrieval-vs-verifier
attribution — the thing the real spec's "report retrieval recall separately"
metric exists to give you — was done here by hand, reading `retrieved` and
`quotes_rejected` off six JSON objects, precisely because there's no harness
to do it automatically yet.

## If this gets built

Priority order, if the step-10 autograder and step-9 harness get picked back
up: (1) implement `factcheck.evaluation.harness` to at least compute the
metrics the README already documents as intended — per-class precision/recall
with bootstrap CIs, dev/test split enforcement — since that's pure aggregation
over data `run_gold.py` already produces, no new model calls required; (2)
only then consider an actual LLM-judge autograder for citation quality, which
needs its own calibration step (agreement against a human-labeled subset)
before its scores mean anything, per `HANDOFF.md`'s original framing of that
step.
