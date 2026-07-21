#!/usr/bin/env python3
"""
compare_judges.py - evaluate the three collected judge outputs and produce the
evidence to decide whether to keep all three or drop any.

GUIDING PRINCIPLE (enforced in the logic below)
------------------------------------------------
These judges interpret contested events with no ground truth. Therefore:
  * Disagreement between judges is DATA, not noise. It is reported (Part 3) as a
    robustness result and is NEVER used as grounds to drop a judge.
  * A judge is a candidate to drop ONLY for QUALITY failures (Part 1): broken
    schema, missing coverage, or wrong self-projection.
  * A judge's own geopolitical lean (Part 2) is MEASURED and reported, never ranked
    or used for selection.
  * Self-family cells (a judge scoring its own model family) are contaminated. They
    are flagged and EXCLUDED from inter-judge agreement; the clean judge (Gemini) is
    treated as authoritative for them. The contaminated pairs come from
    judge_outputs/run_summary.json (`self_family_pairs`).

The recommendation in Part 4 is ADVISORY. It ends with "override as you see fit."

INPUTS
------
  judge_outputs/*.json         one file per judge (each with a top-level `records`).
  judge_outputs/run_summary.json  (optional) supplies `self_family_pairs`.

OUTPUTS
-------
  stdout                        all four parts as readable tables.
  judge_outputs/comparison_report.json   every metric (required deliverable).
  judge_outputs/coverage_heatmap.png     (optional) judged-model x dimension coverage.
  judge_outputs/agreement_by_dimension.png (optional) inter-judge agreement bars.

USAGE
-----
  python compare_judges.py [--judge-dir judge_outputs] [--n-events 150]

DEPENDENCIES: standard library + numpy; matplotlib optional (charts skipped if absent).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Rubric constants
# ---------------------------------------------------------------------------

CANONICAL_DIMENSIONS = [
    "blame_attribution", "moral_framing", "perceived_legitimacy",
    "emotional_tone", "factual_credibility", "motivation_attribution",
]

CANONICAL_JUDGED_MODELS = ["claude", "deepseek", "grok", "gpt-4.1", "llama"]

# Canonical flag vocabulary (rubric A5 + coherence values).
CANONICAL_FLAGS = {
    "collapsed", "saturated", "refusal", "favored_party_zero_intensity",
    "schema_broken", "unprojectable", "actor_not_declared", "merged_actor",
    "label_lean_mismatch", "mind_reading_asymmetry",
    "unsupported", "understated", "supported",
}

# Fields every record is expected to carry (rubric OUTPUT.1).
REQUIRED_FIELDS = [
    "model", "event_id", "conflict_id", "dimension", "dimension_role",
    "signed_score", "bloc_signed_score", "classified_frame", "flags",
]
# Of those, the ones that must be non-null for an event-level record to be usable.
CRITICAL_NONNULL = ["model", "event_id", "dimension", "signed_score", "bloc_signed_score"]

# Quality thresholds (candidate-to-drop). Disagreement is NOT among them.
SCHEMA_MIN = 0.90         # >=90% records well-formed
COVERAGE_MIN = 0.80       # >=80% of the expected event-level matrix
PROJECTION_MAX_MISMATCH = 0.10  # <=10% projection errors among derivable records
SATURATE_HI = 0.40        # >40% of intensities at 5 -> saturation flag
FLATLINE_HI = 0.60        # >60% at 0 -> flatline flag


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def slugify(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id)


# ---------------------------------------------------------------------------
# Load judges
# ---------------------------------------------------------------------------

def load_judges(judge_dir: Path):
    """Return {slug: {'model_id', 'records', 'raw'}} for every judge file."""
    skip = {"run_summary.json", "comparison_report.json"}
    judges = {}
    for path in sorted(judge_dir.glob("*.json")):
        if path.name in skip:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log(f"WARNING: {path.name} did not parse as JSON ({exc}); recorded as schema failure.")
            data = None
        slug = path.stem
        judges[slug] = {
            "model_id": slug.replace("_", "/", 1),  # best-effort inverse of slugify
            "records": (data or {}).get("records", []) if isinstance(data, dict) else [],
            "parsed_ok": data is not None,
            "top_keys": list(data.keys()) if isinstance(data, dict) else None,
        }
    return judges


def load_self_family(judge_dir: Path):
    """Return (pairs, contaminating_judge_slugs). pairs = [{judge, judged_model}]."""
    path = judge_dir / "run_summary.json"
    if not path.exists():
        return [], set()
    try:
        pairs = json.loads(path.read_text(encoding="utf-8")).get("self_family_pairs", [])
    except json.JSONDecodeError:
        return [], set()
    contaminating = {slugify(p["judge"]) for p in pairs}
    return pairs, contaminating


def is_self_family_cell(judge_slug, judged_model, pairs):
    for p in pairs:
        if slugify(p["judge"]) == judge_slug and p["judged_model"] == judged_model:
            return True
    return False


# ---------------------------------------------------------------------------
# Part 1 - Quality
# ---------------------------------------------------------------------------

def project_expected(rec):
    """Re-derive signed_score from favored_party + poles + raw_intensity (rubric A1).
    Returns (expected_or_None, derivable_bool). Only the unambiguous cases are
    checked; third-actor/half-magnitude cases are treated as non-derivable."""
    fp = rec.get("favored_party")
    ri = rec.get("raw_intensity")
    pos = rec.get("pole_positive")
    neg = rec.get("pole_negative")
    if ri is None or not isinstance(ri, (int, float)):
        return None, False
    if ri == 0:
        return 0, True
    if isinstance(fp, str) and fp.lower() == "none":
        return 0, True
    if fp is None:
        return None, False
    if pos is not None and fp == pos:
        return ri, True
    if neg is not None and fp == neg:
        return -ri, True
    # favored_party is a third actor (US/UN/patron) -> half-magnitude/null: skip.
    return None, False


def quality_for_judge(slug, judge, n_events, pairs):
    records = judge["records"]
    n = len(records)
    q = {"judge": slug, "model_id": judge["model_id"], "n_records": n,
         "parsed_ok": judge["parsed_ok"]}

    # --- 1. Schema validity ---
    malformed = 0   # missing a required KEY
    incomplete = 0  # keys present but a critical field is null
    for r in records:
        if not isinstance(r, dict) or any(k not in r for k in REQUIRED_FIELDS):
            malformed += 1
            continue
        if any(r.get(k) is None for k in CRITICAL_NONNULL):
            incomplete += 1
    well_formed = n - malformed - incomplete
    schema_valid = (well_formed / n) if n else 0.0
    q["schema"] = {
        "malformed": malformed, "incomplete_null_critical": incomplete,
        "well_formed": well_formed, "schema_valid_pct": round(100 * schema_valid, 1),
    }

    # --- 2. Coverage of the event-level (model x event x dimension) matrix ---
    event_cells = {(r.get("model"), r.get("event_id"), r.get("dimension"))
                   for r in records if isinstance(r.get("event_id"), int)}
    expected = len(CANONICAL_JUDGED_MODELS) * n_events * len(CANONICAL_DIMENSIONS)
    coverage = len(event_cells) / expected if expected else 0.0
    per_model_cov = {}
    for m in CANONICAL_JUDGED_MODELS:
        present = len({c for c in event_cells if c[0] == m})
        per_model_cov[m] = round(100 * present / (n_events * len(CANONICAL_DIMENSIONS)), 3)
    dims_used = sorted({r.get("dimension") for r in records if r.get("dimension")})
    dims_skipped = [d for d in CANONICAL_DIMENSIONS if d not in dims_used]
    events_covered = sorted({r.get("event_id") for r in records if isinstance(r.get("event_id"), int)})
    null_event_rows = sum(1 for r in records if not isinstance(r.get("event_id"), int))
    q["coverage"] = {
        "event_cells_present": len(event_cells), "expected_cells": expected,
        "coverage_pct": round(100 * coverage, 3),
        "per_judged_model_pct": per_model_cov,
        "distinct_events_covered": len(events_covered),
        "dimensions_used": dims_used, "dimensions_skipped": dims_skipped,
        "summary_or_null_event_rows": null_event_rows,
    }

    # --- 3a. Flag vocabulary compliance ---
    flag_counts = {}
    records_with_noncanon = 0
    for r in records:
        fl = r.get("flags") or []
        has_noncanon = False
        for f in fl:
            flag_counts[f] = flag_counts.get(f, 0) + 1
            if f not in CANONICAL_FLAGS:
                has_noncanon = True
        if has_noncanon:
            records_with_noncanon += 1
    noncanon = sorted(set(flag_counts) - CANONICAL_FLAGS)
    q["flags"] = {
        "distinct_flags": sorted(flag_counts),
        "noncanonical_flags": noncanon,
        "noncanonical_flag_count": len(noncanon),
        "records_carrying_noncanonical": records_with_noncanon,
    }

    # --- 3b. Projection sanity ---
    checked = mismatch = 0
    examples = []
    for r in records:
        exp, ok = project_expected(r)
        if not ok:
            continue
        checked += 1
        got = r.get("signed_score")
        if got is None or abs(float(got) - exp) > 1e-9:
            mismatch += 1
            if len(examples) < 5:
                examples.append({"model": r.get("model"), "event_id": r.get("event_id"),
                                 "dimension": r.get("dimension"), "expected": exp, "got": got})
    q["projection"] = {
        "derivable_records": checked, "mismatches": mismatch,
        "mismatch_pct": round(100 * mismatch / checked, 1) if checked else None,
        "examples": examples,
    }

    # --- 4. Calibration (raw_intensity distribution) ---
    intensities = [r.get("raw_intensity") for r in records
                   if isinstance(r.get("raw_intensity"), (int, float))]
    dist = {i: intensities.count(i) for i in range(6)}
    ni = len(intensities)
    pct5 = (dist[5] / ni) if ni else 0.0
    pct0 = (dist[0] / ni) if ni else 0.0
    q["calibration"] = {
        "n_scored": ni, "distribution": dist,
        "pct_at_5": round(100 * pct5, 1), "pct_at_0": round(100 * pct0, 1),
        "saturates": pct5 > SATURATE_HI, "flatlines": pct0 > FLATLINE_HI,
    }

    # --- Verdict (quality only; disagreement never counts) ---
    reasons = []
    if not judge["parsed_ok"]:
        reasons.append("output did not parse as JSON")
    if schema_valid < SCHEMA_MIN:
        reasons.append(f"schema validity {100*schema_valid:.1f}% < {100*SCHEMA_MIN:.0f}%")
    if coverage < COVERAGE_MIN:
        reasons.append(f"coverage {100*coverage:.3f}% < {100*COVERAGE_MIN:.0f}%")
    if q["projection"]["mismatch_pct"] is not None and q["projection"]["mismatch_pct"] > 100 * PROJECTION_MAX_MISMATCH:
        reasons.append(f"projection mismatch {q['projection']['mismatch_pct']:.1f}% > {100*PROJECTION_MAX_MISMATCH:.0f}%")
    q["quality_pass"] = len(reasons) == 0
    q["fail_reasons"] = reasons
    return q


# ---------------------------------------------------------------------------
# Part 2 - Each judge's own lean
# ---------------------------------------------------------------------------

def lean_for_judge(judge):
    records = judge["records"]

    def mean_of(vals):
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        return round(float(np.mean(vals)), 3) if vals else None

    overall = mean_of([r.get("bloc_signed_score") for r in records])
    per_dim = {}
    for d in CANONICAL_DIMENSIONS:
        per_dim[d] = mean_of([r.get("bloc_signed_score") for r in records if r.get("dimension") == d])
    return {"overall_mean_bloc_signed": overall, "per_dimension": per_dim,
            "n_scored": sum(1 for r in records if isinstance(r.get("bloc_signed_score"), (int, float)))}


# ---------------------------------------------------------------------------
# Part 3 - Inter-judge agreement
# ---------------------------------------------------------------------------

def krippendorff_alpha_interval(unit_values):
    """Interval-metric Krippendorff's alpha via the coincidence matrix.
    unit_values: list of lists (each list = the >=2 ratings for one unit)."""
    units = [u for u in unit_values if len(u) >= 2]
    if len(units) < 2:
        return None
    # Coincidence counts o[(a,b)] over ordered pairs within units, weighted 1/(m-1).
    from collections import defaultdict
    o = defaultdict(float)
    for u in units:
        m = len(u)
        w = 1.0 / (m - 1)
        for a in u:
            for b in u:
                if a is not b:
                    o[(a, b)] += w
    values = [v for u in units for v in u]
    marg = defaultdict(float)
    for (a, b), c in o.items():
        marg[a] += c
    n = sum(marg.values())
    if n <= 1:
        return None
    do = sum(c * (a - b) ** 2 for (a, b), c in o.items()) / n
    # Expected disagreement over all value pairs weighted by marginals.
    keys = list(marg)
    de = 0.0
    for a in keys:
        for b in keys:
            de += marg[a] * marg[b] * (a - b) ** 2
    de = de / (n * (n - 1))
    if de == 0:
        return 1.0 if do == 0 else None
    return round(1 - do / de, 4)


def inter_judge_agreement(judges, pairs):
    """Build the shared-cell table (excluding self-family cells) and compute
    agreement stats overall and per dimension."""
    # cell -> {judge_slug: record}
    cell_map = {}
    self_family_excluded = 0
    for slug, judge in judges.items():
        for r in judge["records"]:
            if not isinstance(r.get("event_id"), int):
                continue
            if r.get("signed_score") is None:
                continue
            model, eid, dim = r.get("model"), r.get("event_id"), r.get("dimension")
            if is_self_family_cell(slug, model, pairs):
                self_family_excluded += 1
                continue
            cell_map.setdefault((model, eid, dim), {})[slug] = r

    shared = {c: v for c, v in cell_map.items() if len(v) >= 2}

    def stats_over(cells):
        pair_diffs, signs_agree, frames_agree, alpha_units = [], 0, 0, []
        for c, jr in cells.items():
            scores = [float(x["signed_score"]) for x in jr.values()]
            alpha_units.append(scores)
            for a, b in itertools.combinations(scores, 2):
                pair_diffs.append(abs(a - b))
            signs = {(1 if s > 0 else -1 if s < 0 else 0) for s in scores}
            if len(signs) == 1:
                signs_agree += 1
            frames = {x.get("classified_frame") for x in jr.values()}
            if len(frames) == 1:
                frames_agree += 1
        ncells = len(cells)
        return {
            "n_cells": ncells,
            "mean_pairwise_abs_diff": round(float(np.mean(pair_diffs)), 3) if pair_diffs else None,
            "sign_agreement_pct": round(100 * signs_agree / ncells, 1) if ncells else None,
            "frame_agreement_pct": round(100 * frames_agree / ncells, 1) if ncells else None,
            "krippendorff_alpha_interval": krippendorff_alpha_interval(alpha_units),
        }

    overall = stats_over(shared)
    per_dim = {}
    for d in CANONICAL_DIMENSIONS:
        cells_d = {c: v for c, v in shared.items() if c[2] == d}
        if cells_d:
            per_dim[d] = stats_over(cells_d)

    ranking = sorted(
        per_dim.items(),
        key=lambda kv: (kv[1]["mean_pairwise_abs_diff"] is None, kv[1]["mean_pairwise_abs_diff"] or 0),
        reverse=True,
    )
    return {
        "n_shared_cells": len(shared),
        "self_family_cells_excluded": self_family_excluded,
        "overall": overall,
        "per_dimension": per_dim,
        "disagreement_ranking": [d for d, _ in ranking],
        "shared_cell_keys": [list(c) for c in sorted(shared, key=lambda x: (str(x[0]), x[1], str(x[2])))],
    }


# ---------------------------------------------------------------------------
# Optional charts
# ---------------------------------------------------------------------------

def make_charts(judges, quality, agreement, judge_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"(charts skipped: matplotlib unavailable - {exc})")
        return []
    written = []
    try:
        # Coverage heatmap: judge x dimension (count of event cells).
        slugs = list(judges)
        grid = np.zeros((len(slugs), len(CANONICAL_DIMENSIONS)))
        for i, slug in enumerate(slugs):
            for r in judges[slug]["records"]:
                if isinstance(r.get("event_id"), int) and r.get("dimension") in CANONICAL_DIMENSIONS:
                    grid[i, CANONICAL_DIMENSIONS.index(r["dimension"])] += 1
        fig, ax = plt.subplots(figsize=(10, 3 + 0.4 * len(slugs)))
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(CANONICAL_DIMENSIONS)))
        ax.set_xticklabels(CANONICAL_DIMENSIONS, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(slugs)))
        ax.set_yticklabels(slugs, fontsize=8)
        for i in range(len(slugs)):
            for j in range(len(CANONICAL_DIMENSIONS)):
                ax.text(j, i, int(grid[i, j]), ha="center", va="center",
                        color="white" if grid[i, j] < grid.max() / 2 else "black", fontsize=8)
        ax.set_title("Event-level cell coverage (count) by judge x dimension")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        p = judge_dir / "coverage_heatmap.png"
        fig.savefig(p, dpi=120); plt.close(fig); written.append(str(p))

        # Agreement bar chart: mean pairwise abs diff per dimension.
        per_dim = agreement["per_dimension"]
        if per_dim:
            dims = list(per_dim)
            vals = [per_dim[d]["mean_pairwise_abs_diff"] or 0 for d in dims]
            ncell = [per_dim[d]["n_cells"] for d in dims]
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(dims, vals, color="#c44")
            for b, nc in zip(bars, ncell):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"n={nc}", ha="center", va="bottom", fontsize=8)
            ax.set_ylabel("mean pairwise |signed_score diff|")
            ax.set_title("Inter-judge disagreement by dimension (higher = less reproducible)")
            plt.xticks(rotation=40, ha="right", fontsize=8)
            fig.tight_layout()
            p = judge_dir / "agreement_by_dimension.png"
            fig.savefig(p, dpi=120); plt.close(fig); written.append(str(p))
    except Exception as exc:
        log(f"(chart generation error, continuing: {exc})")
    return written


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def build_recommendation(judges, quality, leans, agreement, pairs, clean_judge):
    lines = []
    failed = [q for q in quality if not q["quality_pass"]]
    passed = [q for q in quality if q["quality_pass"]]

    # (a) quality
    if failed:
        lines.append("QUALITY: the following judges FAIL a quality gate (candidate to drop):")
        for q in failed:
            lines.append(f"  - {q['model_id']}: " + "; ".join(q["fail_reasons"]))
    else:
        lines.append("QUALITY: all three judges pass the quality gates.")

    # If coverage fails for EVERY failing judge, that is a systemic collection problem.
    coverage_fails_all = bool(failed) and all(
        any("coverage" in r for r in q["fail_reasons"]) for q in failed)
    if failed and len(failed) == len(quality):
        lines.append("  NOTE: ALL judges fail - this is a corpus-wide collection/prompt-fidelity")
        lines.append("        problem (the judges summarized instead of enumerating per-event rows),")
        lines.append("        NOT a reason to prefer one judge over another. Dropping one fixes nothing.")
        if coverage_fails_all:
            lines.append("        Recommended fix: re-collect with per-event enumeration enforced")
            lines.append("        (batch by conflict/event group, higher max_tokens) before analysis.")

    # (b) keep-all guidance
    if not failed:
        ov = agreement["overall"]["mean_pairwise_abs_diff"]
        lines.append(f"KEEP ALL THREE; use inter-judge agreement as a robustness measure "
                     f"(overall mean pairwise |diff| = {ov}).")

    # (c) self-family
    lines.append("SELF-FAMILY (trust the clean judge here):")
    for p in pairs:
        lines.append(f"  - {p['judge']} scoring '{p['judged_model']}' is contaminated -> "
                     f"trust {clean_judge} for those cells.")
    lines.append(f"  ({agreement['self_family_cells_excluded']} self-family cell(s) were excluded "
                 "from agreement stats.)")

    # (d) worst dimension -- but only if there is enough overlap to mean anything.
    ranking = agreement["disagreement_ranking"]
    if agreement["n_shared_cells"] < 10:
        lines.append("RUBRIC WEAKNESS: inter-judge agreement is NOT meaningfully computable - the "
                     f"judges share only {agreement['n_shared_cells']} overlapping event cell(s) "
                     f"(across {len(ranking)} dimension(s), all at a handful of events). The "
                     "reproducibility question cannot be answered from these outputs; that near-zero "
                     "overlap is itself the headline finding to report in the paper.")
    elif ranking:
        worst = ranking[0]
        wd = agreement["per_dimension"][worst]
        lines.append(f"RUBRIC WEAKNESS: worst inter-judge agreement is on '{worst}' "
                     f"(mean pairwise |diff| = {wd['mean_pairwise_abs_diff']}, n_cells = {wd['n_cells']}) "
                     "- flag this dimension as least reproducible in the paper.")
    else:
        lines.append("RUBRIC WEAKNESS: inter-judge agreement is NOT computable - the judges share "
                     f"no overlapping event cells. That near-zero overlap is itself the finding.")

    lines.append("")
    lines.append("This recommendation is advisory - override as you see fit.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_part1(quality):
    print("\n" + "=" * 80)
    print("PART 1 - QUALITY per judge (the ONLY legitimate grounds to drop)")
    print("=" * 80)
    hdr = f"{'judge':<30} {'schema%':>8} {'cover%':>8} {'proj_mm%':>9} {'sat/flat':>9} {'VERDICT':>8}"
    print(hdr); print("-" * len(hdr))
    for q in quality:
        proj = q["projection"]["mismatch_pct"]
        proj_s = "n/a" if proj is None else f"{proj}"
        cal = q["calibration"]
        satflat = ("SAT" if cal["saturates"] else "") + ("FLAT" if cal["flatlines"] else "") or "-"
        verdict = "PASS" if q["quality_pass"] else "FAIL"
        print(f"{q['model_id']:<30} {q['schema']['schema_valid_pct']:>8} "
              f"{q['coverage']['coverage_pct']:>8} {proj_s:>9} {satflat:>9} {verdict:>8}")
    print()
    for q in quality:
        print(f"* {q['model_id']}  ({q['n_records']} records)")
        s, c = q["schema"], q["coverage"]
        print(f"    schema : well_formed={s['well_formed']} malformed={s['malformed']} "
              f"incomplete(null-critical)={s['incomplete_null_critical']} -> {s['schema_valid_pct']}%")
        print(f"    coverage: {c['event_cells_present']}/{c['expected_cells']} cells "
              f"({c['coverage_pct']}%); distinct events={c['distinct_events_covered']}; "
              f"summary/null-event rows={c['summary_or_null_event_rows']}")
        print(f"             dims used={c['dimensions_used']}")
        print(f"             dims skipped entirely={c['dimensions_skipped']}")
        print(f"             per-judged-model coverage%={c['per_judged_model_pct']}")
        fl = q["flags"]
        print(f"    flags  : {len(fl['distinct_flags'])} distinct; "
              f"{fl['noncanonical_flag_count']} NON-canonical -> {fl['noncanonical_flags']}")
        pj = q["projection"]
        print(f"    project: derivable={pj['derivable_records']} mismatches={pj['mismatches']} "
              f"({pj['mismatch_pct']}%)" + (f"  e.g. {pj['examples'][0]}" if pj['examples'] else ""))
        cal = q["calibration"]
        print(f"    calib  : n={cal['n_scored']} dist={cal['distribution']} "
              f"@5={cal['pct_at_5']}% @0={cal['pct_at_0']}% "
              f"{'SATURATES ' if cal['saturates'] else ''}{'FLATLINES' if cal['flatlines'] else ''}")
        print(f"    VERDICT: {'PASS' if q['quality_pass'] else 'FAIL -> ' + '; '.join(q['fail_reasons'])}")
        print()


def print_part2(leans):
    print("=" * 80)
    print("PART 2 - each judge's OWN lean  (for reporting, NOT for selection)")
    print("  (+ = toward western_nato ; - = toward global_south)")
    print("=" * 80)
    dims = CANONICAL_DIMENSIONS
    hdr = f"{'judge':<30} {'overall':>8} " + " ".join(f"{d[:6]:>7}" for d in dims)
    print(hdr); print("-" * len(hdr))
    for slug, L in leans.items():
        row = f"{slug:<30} {str(L['overall_mean_bloc_signed']):>8} "
        row += " ".join(f"{str(L['per_dimension'][d]):>7}" for d in dims)
        print(row)
    print("  (values are mean bloc_signed_score; None = judge produced no scored rows for that dimension)")
    print()


def print_part3(agreement):
    print("=" * 80)
    print("PART 3 - INTER-JUDGE agreement  (the actual finding; disagreement is DATA)")
    print("=" * 80)
    print(f"shared cells (present in >=2 judges, self-family excluded): {agreement['n_shared_cells']}")
    print(f"self-family cells excluded: {agreement['self_family_cells_excluded']}")
    ov = agreement["overall"]
    print(f"OVERALL: n_cells={ov['n_cells']} mean_pairwise|diff|={ov['mean_pairwise_abs_diff']} "
          f"sign_agree={ov['sign_agreement_pct']}% frame_agree={ov['frame_agreement_pct']}% "
          f"alpha(interval)={ov['krippendorff_alpha_interval']}")
    if agreement["n_shared_cells"] < 10:
        print("  !! WARNING: overlap is tiny - these statistics are DESCRIPTIVE ONLY, not reliable.")
    print()
    if agreement["per_dimension"]:
        hdr = f"{'dimension':<26} {'n_cells':>7} {'mean|diff|':>10} {'sign%':>7} {'frame%':>7} {'alpha':>7}"
        print(hdr); print("-" * len(hdr))
        for d in agreement["disagreement_ranking"]:
            s = agreement["per_dimension"][d]
            print(f"{d:<26} {s['n_cells']:>7} {str(s['mean_pairwise_abs_diff']):>10} "
                  f"{str(s['sign_agreement_pct']):>7} {str(s['frame_agreement_pct']):>7} "
                  f"{str(s['krippendorff_alpha_interval']):>7}")
        print(f"\n  disagreement ranking (worst first): {agreement['disagreement_ranking']}")
    else:
        print("  No dimension has >=2 judges sharing any event cell -> agreement is uncomputable.")
    print()


def print_part4(recommendation):
    print("=" * 80)
    print("PART 4 - RECOMMENDATION  (advisory; you decide)")
    print("=" * 80)
    print(recommendation)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Compare three judge outputs (quality, lean, agreement).")
    ap.add_argument("--judge-dir", default="judge_outputs")
    ap.add_argument("--n-events", type=int, default=150,
                    help="Event count for the coverage denominator (default 150).")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    judge_dir = Path(args.judge_dir).resolve()
    if not judge_dir.exists():
        raise SystemExit(f"ERROR: judge dir not found: {judge_dir}")

    judges = load_judges(judge_dir)
    if not judges:
        raise SystemExit(f"ERROR: no judge JSON files found in {judge_dir}")
    pairs, contaminating = load_self_family(judge_dir)

    # The clean judge = a judge that never appears as a contaminating .judge.
    clean_candidates = [j["model_id"] for slug, j in judges.items() if slug not in contaminating]
    clean_judge = next((m for m in clean_candidates if "gemini" in m.lower()),
                       clean_candidates[0] if clean_candidates else "(none)")

    log(f"Loaded {len(judges)} judges: {[j['model_id'] for j in judges.values()]}")
    log(f"Self-family pairs: {pairs}")
    log(f"Clean (authoritative-for-self-family) judge: {clean_judge}")

    quality = [quality_for_judge(slug, judges[slug], args.n_events, pairs) for slug in judges]
    leans = {slug: lean_for_judge(judges[slug]) for slug in judges}
    agreement = inter_judge_agreement(judges, pairs)
    recommendation = build_recommendation(judges, quality, leans, agreement, pairs, clean_judge)

    charts = [] if args.no_charts else make_charts(judges, quality, agreement, judge_dir)

    # --- stdout: four parts ---
    print_part1(quality)
    print_part2(leans)
    print_part3(agreement)
    print_part4(recommendation)
    if charts:
        print(f"charts written: {charts}\n")

    # --- required deliverable: comparison_report.json ---
    report = {
        "judges": [j["model_id"] for j in judges.values()],
        "self_family_pairs": pairs,
        "clean_judge_authoritative_for_self_family": clean_judge,
        "n_events_denominator": args.n_events,
        "part1_quality": quality,
        "part2_judge_lean": leans,
        "part3_inter_judge_agreement": agreement,
        "part4_recommendation": recommendation,
        "charts": charts,
    }
    out = judge_dir / "comparison_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Full metrics written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
