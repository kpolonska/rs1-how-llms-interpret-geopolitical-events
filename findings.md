# Model narrative-bias study — findings

_Local analysis of `judge_opus_4.8.json` (primary, Claude Opus 4.8) and the embedded `second_judge_gpt_5_6` (GPT-5.6-sol). Descriptive, not inferential._

## Guardrails

- **Numeric-score agreement is not reliability.** `signed_score` is a deterministic projection of (favored_party, raw_intensity, poles); the 100% cross-judge match is an arithmetic **validity check**. Real reliability = flag agreement (**45.2%**) and frame agreement (**90.5%**).
- **stimulus_suspect events excluded from all lean claims** — 17 events: [0, 40, 54, 57, 58, 59, 62, 66, 68, 73, 77, 85, 87, 89, 90, 97, 108].
- **Self-family cells flagged.** claude is scored by an Opus judge (Opus×claude, self_family=true); hatched in every figure showing claude's lean.
- **No model met the consistent-lean threshold** (|mean|>=1.5, sd<=1.5, >=70% same sign). All five are `volatile`: leanings present but volatile, not consistent.
- **n=45 events (21 two-judge).** SD error bars show spread, not significance.

> statistical_note: With five judged models and few events per conflict, SDs and consistent_lean are descriptive, not inferential - error bars show spread, not significance. Every raw per-event signed score is retained in records, so bootstrap CIs can be added later without re-judging.

## Figures (core set)

**F1 — Model × dimension lean.** Mean `bloc_signed_score` per model across the four comparison dimensions, stimulus_suspect excluded. Every SD bar dwarfs its mean — the visual signature of volatile, not consistent, leanings. grok is the lone slight western tilt; the others lean global-south. claude bars are hatched (self-scored).

**F2 — Lean by conflict.** Blame-attribution mean per model per conflict. Leans are conflict-specific: grok/gpt-4.1 tilt western on Ukraine but most models swing global-south on Lebanon/Gaza. Cell counts are small — read direction, not magnitude.

**F3 — Inter-model disagreement.** Dimensions ranked by mean pairwise |score diff|. **blame_attribution** is the most-contested axis (**2.27**), driven by causal-start-point and actor-selection divergence.

**F4 — Consistency.** Lean strength vs. per-event noise, marker size = sign flips. Every model sits above the noise=lean line, so per-event noise ≈ or exceeds the average lean. All five are `volatile` — the 'leanings exist but aren't stable' finding.

**F6 — Two-judge reliability.** Frame agreement **90.5%**, flag agreement **45.2%**; the deterministic numeric match is shown only as a separate validity-check bar. All 10 frame disagreements trace to llama (non-standard declared labels).

**F7 — Asymmetric credibility (mechanism).** Mean credibility granted to western vs. global-south pole claims (credible=+1, propagandistic=−1). deepseek doubts western claims while crediting global-south — a concrete mechanism for its lean; grok is the only model crediting western more. (Intended `hedge_counts` is near-empty — 4 corpus-wide — so `credibility_verdict` is used instead.)

_Not shown (available on request): declared-vs-revealed frame mismatch (label_lean_mismatch, stimulus-excluded: claude=0, deepseek=12, grok=0, gpt-4.1=0, llama=36) and emotional-tone-vs-blame scatter._

## Per-model lean (comparison dimensions, stimulus_suspect excluded)

| model | blame_attribution | moral_framing | perceived_legitimacy | emotional_tone | reliability | self_scored |
| --- | --- | --- | --- | --- | --- | --- |
| claude | -0.5 | -0.39 | -0.29 | -0.43 | volatile | yes |
| deepseek | -2.82 | -1.57 | -1.61 | -1.82 | volatile |  |
| grok | 1.0 | 0.61 | 0.96 | -0.18 | volatile |  |
| gpt-4.1 | -1.07 | -1.11 | -0.79 | -1.46 | volatile |  |
| llama | -2.04 | -0.89 | -0.75 | -1.82 | volatile |  |

## Most defensible findings

1. **Blame attribution is the most-contested dimension** (mean pairwise |diff| 2.27) — a clean, judge-independent structural result.
2. **Leanings are present but volatile.** No model met the consistent-lean threshold; all five are `volatile`, so bloc means are spreads, not point estimates.
3. **Judges agree on frames, only moderately on flags** (90.5% vs. 45.2%); the ~100% numeric match is a deterministic validity check. claude cells are self-scored under the Opus judge and carry that caveat.