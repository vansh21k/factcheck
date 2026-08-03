# Spec: Grounded claim verification over an arXiv corpus

Status: ready-for-agent

Design reference: https://claude.ai/code/artifact/d5c9d979-080d-4504-add7-46123deb4a5d

## Problem Statement

I need to decide whether a factual claim is true, and the only admissible authority is a fixed corpus of research paper abstracts. Doing this by hand means reading everything that might be relevant, holding several passages in my head at once, and staying honest about the difference between "the corpus disagrees with this" and "the corpus simply does not say."

An LLM answers instantly and is the wrong tool used naively, because it will answer from training data rather than from the corpus, and it will produce a fluent, confident, well-cited-looking verdict whose citations do not exist. That failure is worse than no answer at all: it is indistinguishable from a correct one at a glance, and it is most convincing precisely where the corpus is silent.

I also cannot tell, from a verdict alone, whether a wrong answer came from the retriever never surfacing the right document or from the verifier misreading a document it did see. Without that attribution I can't improve the system — I can only re-roll it.

Finally, this must not be a design that only works because today's corpus is fifty abstracts. The same shape has to hold when the corpus is several orders of magnitude larger, and I need to know now which parts would have to change.

## Solution

A command-line system in three programs, split at the network boundary and the index boundary:

- **`fetch`** pulls the corpus from the arXiv API once and stores raw documents.
- **`index`** turns stored documents into retrieval structures with no network access, writing a manifest that records exactly how it was built.
- **`verify`** loads a prebuilt index and runs interactively: type a claim, get a verdict of *supported*, *contradicted*, or *unknown*, together with the verbatim passages that justify it and the documents they came from.

Groundedness is enforced by code, not by prompting. Every quoted span the model returns is mechanically checked to be literally present in the document it cites; anything that fails is discarded, and a claim with no surviving evidence returns *unknown*. A prompt regression can therefore make the system less useful, but it cannot make it ungrounded.

Every stage — query expansion, retrieval, fusion, reranking, verification, auditing, aggregation — sits behind an interface, and a tunable config selects and parameterizes them. An evaluation harness runs a hand-labeled claim set through the identical entry point the CLI uses, scoring retrieval and verification separately so that a wrong verdict points at a subsystem rather than at "the system".

## User Stories

**Corpus ingest**

1. As an engineer, I want a single command to fetch every paper in the corpus by ID, so that I do not assemble the corpus by hand.
2. As an engineer, I want each document to carry its title, authors, published date, updated date, and abstract text, so that evidence can be attributed to a real, citable paper.
3. As an engineer, I want the fetch step to batch IDs into as few API requests as the service allows, so that ingest is fast and polite.
4. As an engineer, I want the fetch step to rate-limit itself between requests, so that I am not throttled or blocked mid-corpus.
5. As an engineer, I want fetch to be resumable, so that a network failure partway through does not force me to start over.
6. As an engineer, I want fetch to report exactly which requested IDs did not resolve, so that a silently short corpus is impossible.
7. As an engineer, I want ingest to fail loudly if it retrieves zero documents, so that a redirect or endpoint change surfaces immediately rather than as universal *unknown* verdicts.
8. As an engineer, I want two versions of the same paper treated as two distinct documents, so that a revision that changes a finding is not silently collapsed into its predecessor.
9. As an engineer, I want the version identifier preserved in every document ID, so that cited evidence is traceable to the exact revision it came from.
10. As an engineer, I want the fetch step separated from the index step, so that re-chunking or changing embedding models never re-hits the network.

**Indexing**

11. As an engineer, I want an index command that performs no network access, so that index builds are reproducible offline.
12. As an engineer, I want indexing to be deterministic given the same documents and config, so that two builds of the same corpus are comparable.
13. As an engineer, I want documents split into chunks with recorded character offsets, so that evidence spans locate precisely within a source document.
14. As an engineer, I want the index to record the exact embedding model used to build it, so that a query-time mismatch is detectable.
15. As an engineer, I want the index to record a content hash of the corpus, so that a stale index can be detected against changed documents.
16. As an engineer, I want the index to record chunker settings and build timestamp, so that I can tell two index builds apart without reading their contents.
17. As an engineer, I want indexing to report document and chunk counts on completion, so that I can sanity-check the build.
18. As an engineer, I want a rebuild to be an explicit flag, so that expensive work never happens as a side effect of something else.

**Retrieval**

19. As an engineer, I want a subset of relevant passages selected before any verification call, so that the design does not depend on the corpus fitting in a context window.
20. As an engineer, I want both lexical and semantic retrieval, so that exact terms like benchmark names and numerals and paraphrased concepts are both reachable.
21. As an engineer, I want retrieval results combined by rank rather than by raw score, so that no per-corpus score normalization has to be tuned.
22. As an engineer, I want a claim expanded into multiple retrieval queries, so that a multi-part claim surfaces evidence for each of its parts.
23. As an engineer, I want query expansion to include a negation form of the claim, so that refuting passages are retrievable and *contradicted* is a reachable verdict.
24. As an engineer, I want a reranking stage present in the pipeline even where it changes little today, so that the precision stage exists at the scale where it matters.
25. As an engineer, I want to cap how many passages reach the verifier, so that context size and cost stay bounded as the corpus grows.
26. As an engineer, I want a minimum similarity threshold on semantic results, so that a claim unrelated to the corpus retrieves nothing rather than the least-irrelevant thing available.
27. As an engineer, I want each retrieved passage tagged with a stable document identifier, so that the verifier can cite it unambiguously.
28. As an engineer, I want the set of documents the verifier saw returned on the result, so that retrieval failures and verification failures are distinguishable.

**Verification and grounding**

29. As an engineer, I want a claim classified as supported, contradicted, or unknown, so that the output is a decision rather than an essay.
30. As an engineer, I want *contradicted* to require evidence asserting the opposite of the claim, so that a claim the corpus is merely silent about is never reported as false.
31. As an engineer, I want *unknown* returned whenever support cannot be established, so that abstention is the default and answering is what must be earned.
32. As an engineer, I want the verifier to see only retrieved passages, so that its own training knowledge cannot leak into a verdict.
33. As an engineer, I want the response schema to make an uncited verdict structurally impossible, so that citation is a requirement of the type rather than a request in a prompt.
34. As an engineer, I want every quoted span mechanically verified as literally present in the document it cites, so that fabricated evidence is caught by code.
35. As an engineer, I want span checking to tolerate whitespace and typographic quote differences, so that trivial formatting variation is not reported as fabrication.
36. As an engineer, I want a claim whose every quote fails validation to return *unknown*, so that hallucinated citations degrade into abstention rather than into a wrong verdict.
37. As an engineer, I want a minimum length on accepted quotes, so that a trivially verbatim fragment cannot satisfy the grounding check while proving nothing.
38. As an engineer, I want a maximum length on accepted quotes, so that quoting an entire abstract does not count as pinpoint evidence.
39. As an engineer, I want each surviving quote independently re-checked against the claim in isolation, so that a genuinely verbatim but irrelevant span is caught.
40. As an engineer, I want that second check to see neither the retrieved context nor the first pass's reasoning, so that it cannot simply ratify the first pass.
41. As an engineer, I want the final verdict decided by a fixed rule over the collected stances, so that arbitration is inspectable code rather than model judgment.
42. As an engineer, I want conflicting stances across two versions of one paper surfaced as a flag with both sides shown, so that a genuine disagreement in the corpus is reported rather than silently resolved.
43. As an engineer, I want every returned evidence item to name its document, its verbatim quote, and its stance, so that I can check the verdict myself in seconds.
44. As an engineer, I want evidence to carry the character offsets of the span, so that I can locate it in the source document exactly.

**Configuration**

45. As an engineer, I want all tunable values in one config object rather than scattered through the code, so that the tunable surface is visible in one place.
46. As an engineer, I want the config immutable for the duration of a run, so that a result always corresponds to exactly one configuration.
47. As an engineer, I want the config hashed and recorded on every evaluation result, so that a number can always be traced back to what produced it.
48. As an engineer, I want config loadable from a file and overridable per invocation, so that experiments do not require code edits.
49. As an engineer, I want index-time settings distinguished from query-time settings in the config type itself, so that I know what requires a rebuild without consulting a document.
50. As an engineer, I want an attempt to vary an index-time setting against a prebuilt index to fail, so that a sweep cannot produce plausible-looking rows that mean nothing.
51. As an engineer, I want to tune how many passages reach the verifier, so that I can trade cost against recall.
52. As an engineer, I want to tune retrieval score thresholds, so that I can trade precision against coverage.
53. As an engineer, I want to tune validator normalization and quote-length bounds, so that I can calibrate strictness without touching the substring requirement itself.
54. As an engineer, I want to disable the second verification pass by config, so that its contribution can be measured rather than assumed.
55. As an engineer, I want to disable query expansion by config, so that its cost and its effect on the contradicted class can be measured.
56. As an engineer, I want prompts versioned as config values, so that a prompt change is as traceable as a threshold change.
57. As an engineer, I want no configuration option that relaxes the requirement for a quote to be present in its document, so that the grounding guarantee is not a tuning parameter.

**Interactive CLI**

58. As an engineer, I want the CLI to load a prebuilt index and then accept claims interactively, so that checking several claims does not re-pay startup cost each time.
59. As an engineer, I want the CLI to print the loaded index's document count, build time, and config hash on startup, so that I always know what I am querying.
60. As an engineer, I want the CLI to refuse to start when the index's embedding model differs from the configured one, so that silently degraded retrieval is impossible.
61. As an engineer, I want the CLI to warn when the index is stale relative to the documents, so that I am not unknowingly querying an old corpus.
62. As an engineer, I want pointing the CLI at a directory with no index to be an error naming the index command, so that an implicit rebuild never hides the cost that dominates at scale.
63. As an engineer, I want a verdict rendered with its evidence, source titles, and dates, so that I can judge the answer without opening another tool.
64. As an engineer, I want per-claim counters for passages retrieved, quotes accepted, quotes rejected, LLM calls, and elapsed time, so that cost and behavior are visible during ordinary use.
65. As an engineer, I want a command that explains the last verdict — the retrieved set, the fusion ranks, each quote's audit stance, and why each rejected span failed — so that I can attribute a bad answer to a stage.
66. As an engineer, I want to change a query-time config value inside the session, so that I can feel the effect of a knob without restarting.
67. As an engineer, I want an attempt to change an index-time value in-session to be refused with the rebuild command named, so that the config boundary is enforced where I actually make the mistake.
68. As an engineer, I want a one-shot mode that verifies a single claim and exits, so that the tool composes into scripts.
69. As an engineer, I want a batch mode that reads claims from a file, so that the evaluation harness drives the exact path the CLI drives.
70. As an engineer, I want a machine-readable output mode, so that results can be post-processed without scraping formatted text.
71. As an engineer, I want a missing API key to be reported clearly at startup, so that the failure is diagnosed in seconds rather than mid-claim.

**Evaluation**

72. As an engineer, I want a hand-labeled set of claims drawn from the corpus, so that verdict quality is measured rather than sampled by impression.
73. As an engineer, I want that set to include claims that are on-topic and plausible but unsupported, so that the hardest class is represented rather than avoided.
74. As an engineer, I want each labeled claim annotated with the document that should be cited, so that retrieval can be scored independently of verdicts.
75. As an engineer, I want retrieval recall reported separately from verdict accuracy, so that I know which subsystem to fix.
76. As an engineer, I want per-class precision and recall, so that a system that never returns *contradicted* is visibly bad rather than averagely good.
77. As an engineer, I want the rate of quotes rejected by span validation tracked over time, so that prompt drift is detected before it reaches verdict quality.
78. As an engineer, I want the abstention rate reported, so that a system optimizing toward always answering *unknown* is caught.
79. As an engineer, I want a grounded automatic grader that judges a verdict only against its own cited quotes, so that unlabeled claims can be scored at volume.
80. As an engineer, I want the automatic grader's agreement with the human labels measured first, so that its own error rate is known before it is trusted.
81. As an engineer, I want confidence intervals reported alongside every metric, so that I do not act on differences the eval set cannot resolve.
82. As an engineer, I want the labeled set split into tuning and held-out portions, so that repeated tuning does not silently overfit the number I report.
83. As an engineer, I want to run the same labeled set across multiple configurations and get a comparison table, so that design choices are measured rather than asserted.
84. As an engineer, I want stage outputs cached by their inputs, so that sweeping a downstream knob does not re-pay for identical upstream calls.

**Extensibility**

85. As an engineer, I want every pipeline stage behind an interface, so that an implementation can be replaced without editing the pipeline.
86. As an engineer, I want a multi-source retriever to itself be a retriever, so that adding a retrieval mechanism is a list entry rather than a structural change.
87. As an engineer, I want rank fusion to be its own replaceable component, so that fusion strategies can be compared rather than assumed.
88. As an engineer, I want index building and index querying to be separate interfaces, so that query-time code cannot reach for build-time behavior.
89. As an engineer, I want the verification entry point to depend only on interfaces, so that tests substitute a scripted model without a network call.
90. As an engineer, I want swapping the lexical or semantic backend for a service-backed one to require no change above the retrieval interface, so that the scale story is a swap rather than a rewrite.

## Implementation Decisions

**Language and dependencies.** Python, in a standalone project outside the existing interview-prep repository. Anthropic SDK for model calls. An off-the-shelf BM25 implementation and a local embedding model, chosen so nothing requires a running external service. No containers.

**Three entry points, split at two boundaries.** `fetch` is the only component permitted network access to arXiv. `index` is pure and offline. `verify` is online and interactive. The split exists to separate offline from online concerns: a system that builds its index at startup has assumed the corpus fits in memory and in a few seconds, which is exactly the assumption the scale requirement forbids.

**Ports.** Every stage is a structural-typing protocol rather than a base class, so a test double is any object with the right method and never a subclass:

```
QueryExpander.expand(claim)                  -> list[Query]
Retriever.search(query, k)                   -> list[ScoredChunk]
Fusion.fuse(list[list[ScoredChunk]], k)      -> list[ScoredChunk]
Reranker.rerank(claim, chunks, k)            -> list[ScoredChunk]
Verifier.adjudicate(claim, passages)         -> RawVerdict
Auditor.audit(claim, evidence)               -> Stance
AggregationPolicy.decide(list[Evidence])     -> Verdict
DocumentStore.get(doc_id)                    -> Document
DocumentStore.iter_chunks()                  -> Iterable[Chunk]
IndexBuilder.build(chunks, out_dir)          -> ManifestEntry   # offline
IndexLoader.load(out_dir, manifest)          -> Retriever       # online
```

*(Shape settled during the design pass; the split of `IndexBuilder` from `IndexLoader` is deliberate — a single interface carrying both would let query-time code call `build`, which is the accidental startup rebuild the program split exists to prevent.)*

**Hybrid retrieval as composition.** The hybrid retriever *is* a retriever, composed of a list of retrievers plus a fusion strategy. "Swap the mechanism" and "go from one retriever to three" are therefore the same operation, and nothing upstream distinguishes them. Fusion is a separate port specifically so reciprocal rank fusion can be compared against alternatives rather than baked in.

**Rank fusion over score blending.** Lexical scores and cosine similarities are incomparable scales; fusing on ranks removes per-corpus normalization tuning entirely.

**Negation-aware query expansion.** Retrieval optimizes for similarity, but fact-checking needs refutation. Without a negation-form query, refuting passages are rarely retrieved and the verifier becomes structurally incapable of returning *contradicted*. This is a correctness decision, not a recall optimization.

**Two model passes with different jobs.** The first adjudicates the claim against all retrieved passages and returns structured evidence. The second re-checks each surviving quote against the claim in isolation, with neither the retrieval context nor the first pass's reasoning available — an auditor shown the first pass's chain largely ratifies it, so the isolation is the entire value. The second pass runs on a smaller model, only over quotes that survived validation, since it is a narrow entailment task.

**Grounding is layered, and only some layers are guarantees.** Constrained input and the entailment audit improve quality. The response schema and span validation are the guarantees that survive prompt drift, model swaps, and temperature changes. Span validation is deliberately *not* a port and not configurable in its core requirement: a swappable guarantee is not a guarantee, and the first convenient substitution is a permissive validator.

**Aggregation is a fixed rule over stances:**

| Stances present | Verdict | Flag |
| --- | --- | --- |
| contradicts | contradicted | — |
| entails | supported | — |
| entails + contradicts | contradicted | conflicting_evidence |
| neither / empty | unknown | — |

The `no surviving evidence -> unknown` short-circuit lives above the aggregation policy, not inside it, so no policy can be written that answers on zero evidence. The policy chooses among verdicts; it does not choose whether evidence was required.

**Version conflicts are surfaced, not resolved.** The 50 requested IDs resolve to 48 distinct papers — two papers appear at two versions each, and their abstracts differ textually under identical titles and published dates. A claim keyed to a revised figure can legitimately draw opposite stances from two versions. Deduplicating hides a real disagreement; letting contradiction silently win misreports a superseded finding. The system reports both sides with version identifiers.

**Config as an immutable value object.** A frozen tree passed into the verification entry point, loadable from file, overridable per invocation, hashed. Index-time fields (chunk size and overlap, embedding model) are structurally distinguished from query-time fields (expansion, thresholds, top-n, validator bounds, model and temperature, prompt version, aggregation policy). Index-time fields are recorded in the manifest; varying one against a prebuilt index is an error, because the failure mode is not a crash but a plausible number.

**Validator knobs bound evidentiary weight, not presence.** Normalization options control *how* text is compared. Minimum and maximum quote length exist because exact-substring checking has holes at both ends: a two-word fragment is verbatim and proves nothing, and an entire abstract quoted as one span is verbatim and pinpoints nothing. There is deliberately no fuzzy-match threshold — it is the knob that converts the grounding guarantee back into a tuning parameter, and paraphrase drift is the auditor's job instead.

**Manifest as the contract between build and query.** Records corpus hash, embedding model, embedding dimension, chunker version, build timestamp, and counts. A dimension mismatch would error loudly; a *different model at the same dimension* fails silently and returns confident nonsense in plausible rank order, so the model identity check is explicit and refusal is at startup.

**Interactive session.** The REPL exposes a command that explains the previous verdict — retrieved set, fusion ranks, per-quote stances, and the reason each rejected span failed — because interactive use is where tuning actually happens and single-claim attribution mirrors what the aggregate metrics provide. A set command applies query-time changes live and refuses index-time changes with the rebuild command named.

**Stage-level caching keyed by stage, input hash, model, and temperature.** Sweeping a downstream knob replays recorded upstream output, making aggregation and validator sweeps nearly free; upstream knobs invalidate everything after them. Sweeps are therefore ordered from the bottom of the pipeline up.

**Result shape.** Verdict, evidence items (document ID, verbatim quote, stance, character span), flags, and the set of retrieved document identifiers. The retrieved set looks like a debugging leak and is the most important field for evaluation: without it, a retriever miss and a verifier misread are indistinguishable.

## Testing Decisions

**What a good test looks like here.** Tests drive the single public verification entry point and assert on the returned result — verdict, evidence, flags. They do not assert on which stage produced what, how many internal calls occurred, or the shape of intermediate structures. Internal stage boundaries are expected to change as retrieval and verification implementations are swapped, and a test that freezes them converts the extensibility this spec asks for into a liability.

**The single seam.** `FactChecker.check(claim)`, constructed with an injected retriever and injected model clients. Tests supply a small fixed corpus and a scripted model that replays predetermined responses, making the suite hermetic, fast, and deterministic with no network access and no API key.

**Grounding tests are the heart of the suite**, because they cover behavior that cannot be verified by inspection:

- A scripted model returning a quote absent from the cited document yields *unknown*, not the model's verdict.
- A model returning a quote attributed to the wrong document yields *unknown*.
- A model returning a quote differing only in whitespace or typographic quote characters is accepted.
- A model returning one valid and one fabricated quote keeps the valid one and drops the other.
- A quote shorter than the configured minimum is rejected despite being literally present.
- A claim on a topic absent from the corpus returns *unknown* rather than a nearest-neighbour verdict.
- Opposing stances across two versions of one paper produce *contradicted* plus the conflict flag with both sides present.
- A claim the corpus is merely silent about returns *unknown* and never *contradicted*.

**Configuration tests.** Varying an index-time value against a prebuilt index is an error. An embedding-model mismatch between config and manifest refuses startup. Disabling the second pass and disabling query expansion each produce a working pipeline, since both are ablation paths the evaluation depends on.

**Ingest tests** run against recorded API responses, not the live service: version-suffixed IDs are preserved distinctly, unresolved IDs are reported, and a response yielding zero documents fails loudly.

**Interface conformance.** Each port has a shared set of behavioral checks that any implementation must pass, so a new retrieval or verification backend is validated against the same contract rather than against its own bespoke tests.

**Evaluation is not a test.** The harness measures quality against the labeled set and hits the real API; it reports metrics rather than passing or failing, and does not run in the unit suite. Metric thresholds are deliberately not asserted, because a labeled set this size cannot resolve small differences and a failing build on statistical noise trains people to ignore the build.

**Prior art.** None in this project — it is greenfield. Conventions established here are the prior art for what follows.

## Out of Scope

- Any graphical interface; the deliverable is command-line only.
- Containers, orchestration, or any externally hosted service dependency.
- Actually operating at the larger scale discussed. The design must not preclude it and the swap points must be identified, but service-backed search and vector indexes are not implemented.
- Full-text ingestion. Abstracts are the document text; PDF or LaTeX body parsing is excluded.
- Corpora other than the fixed list of arXiv identifiers, and any live or incremental corpus updating.
- Multi-hop verification requiring evidence combined across documents to reach a conclusion neither states alone.
- Numeric or unit-aware reasoning beyond what the model does unaided.
- Persistence of verification history, user accounts, authentication, or concurrent access.
- Automated hyperparameter search. The config is sweepable and the harness compares configurations; choosing what to sweep stays manual.
- Fine-tuning or training any model.
- Latency and throughput optimization beyond the caching described.

## Further Notes

**Verified against the live API during design.** The `http` arXiv endpoint answers 301 with an empty body. Without following redirects over HTTPS, ingest silently yields zero documents and every claim returns *unknown* — a failure that reads as appropriate conservatism rather than a broken corpus. This is the reason ingest asserts a document count before the system will answer anything.

**Confirmed corpus shape.** 50 requested identifiers, 48 distinct papers. Two identifiers appear at two versions each, and a diff of one pair showed genuinely different abstract text under an identical title and published date. This was discovered rather than assumed, and it is the origin of the conflict flag.

**The labeled set is the real bottleneck.** Thirty claims cannot resolve a three-point accuracy difference — that is roughly one claim flipping. A tunable config plus a small labeled set is an efficient machine for overfitting, which is why confidence intervals and a held-out split are specified rather than optional. The correct fix is more labeled claims, not a smarter search strategy, and stating which knob *would* matter given a larger set is a better answer than a leaderboard built on thirty examples.

**Abstraction has a genuine cost at this size.** Eleven protocols over a fifty-document corpus is more indirection than today's problem needs. The defense is that the ports were chosen from the axes the task names — swappable retrieval, swappable verification, a corpus orders of magnitude larger — and that everything outside those axes stays concrete. The standing test: a port that never earns a row in an ablation table is speculative generality and should be collapsed.

**Recurring failure class across this design.** The dangerous failures here are not crashes. A silent redirect, a same-dimension embedding-model mismatch, an index-time knob swept against a stale index, and a two-word verbatim quote all produce output that looks entirely reasonable. Each is met with an explicit assertion at the boundary rather than a comment, because none of them is detectable downstream by reading verdicts.

**Anticipated first weakness.** The most likely early failure is under-retrieval of refuting evidence, showing up as high accuracy overall and poor recall specifically on *contradicted*. Query expansion is the designed mitigation and the first thing to measure — which is precisely why the metrics are per-class rather than aggregate.
