#!/usr/bin/env python3
"""
analyze_bias.py  --  local analysis + clean matplotlib figures for the model
narrative-bias study.

Pure local analysis. NO API calls, NO API key, NO network.

Inputs (read from ./ or ./final_judge/):
    judge_opus_4.8.json      -- PRIMARY judge (Claude Opus 4.8)
    judge_gpt_5.6-sol.json   -- standalone GPT-5.6 backup (embedded second_judge is truth)

Outputs:
    figures/*.png            -- F1, F2, F3, F4, F6, F7 (core set), ~150 dpi
    findings.md              -- written summary; all caveats live here, not on the plots
    (stdout)                 -- startup diagnostics + per-model lean table

Analytical guardrails (kept visible on figures where substantive, stated in findings):
  1. ~100% numeric cross-judge match = deterministic VALIDITY CHECK, not reliability.
     Real reliability = flag agreement (~45%) + frame agreement (~90.5%).
  2. stimulus_suspect events excluded from every lean figure (read from the data).
  3. self-family cells (Opus x claude) hatched wherever claude's lean is shown.
  4. Descriptive, not inferential (n=45; 21 two-judge). SD error bars, no sig stars.
  5. No model met the consistent-lean threshold; all five are 'volatile'.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
OPUS_NAMES = ["judge_opus_4.8.json", "judge_opus_4_8.json"]
GPT_NAMES = ["judge_gpt_5.6-sol.json", "judge_gpt_5_6-sol.json"]
SEARCH_DIRS = [HERE, os.path.join(HERE, "final_judge")]


def _find(names):
    for d in SEARCH_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(f"Could not find any of {names} in {SEARCH_DIRS}")


# ---------------------------------------------------------------------------
# Palette + clean style
# ---------------------------------------------------------------------------
# Colorblind-safe (Okabe-Ito), one color per model, reused across every figure.
MODEL_COLORS = {
    "claude":   "#0072B2",
    "deepseek": "#E69F00",
    "grok":     "#009E73",
    "gpt-4.1":  "#CC79A7",
    "llama":    "#D55E00",
}
SELF_HATCH = "///"
ACCENT = "#D55E00"     # highlight
NEUTRAL = "#7F9BB3"    # de-emphasised bars
INK = {"title": "#1a1a1a", "sub": "#6b6b6b", "muted": "#8a8a8a"}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK["title"],
        "axes.labelcolor": "#333333",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#BFBFBF",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": "#E7E7E7",
        "grid.linewidth": 0.8,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
    })


def titles(fig, title, subtitle=None, y=0.985, sub_dy=0.052):
    fig.suptitle(title, y=y, color=INK["title"])
    if subtitle:
        fig.text(0.5, y - sub_dy, subtitle, ha="center", va="top",
                 fontsize=8.5, color=INK["sub"])


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Loading + config (everything driven from the data / meta, not hardcoded)
# ---------------------------------------------------------------------------
def load_data():
    op, gp = _find(OPUS_NAMES), _find(GPT_NAMES)
    with open(op, encoding="utf-8") as f:
        opus = json.load(f)
    with open(gp, encoding="utf-8") as f:
        gpt_backup = json.load(f)
    return opus, gpt_backup, op, gp


def get_config(opus):
    meta = opus["meta"]
    prim = opus["primary_evaluation"]
    role_by_dim = {r["dimension"]: r["dimension_role"] for r in prim["records"]}
    ordered = [d for d in meta.get("dimensions", []) if d in role_by_dim]
    for d in role_by_dim:
        if d not in ordered:
            ordered.append(d)
    comparison = [d for d in ordered if role_by_dim[d] == "comparison"]
    mechanism = [d for d in ordered if role_by_dim[d] == "mechanism"]
    stim = sorted(e["event_id"] for e in prim["event_axis"]
                  if e.get("stimulus_check", {}).get("stimulus_suspect"))
    return {
        "judged_models": meta["judged_models"],
        "comparison_dims": comparison,
        "mechanism_dims": mechanism,
        "all_dims": comparison + mechanism,
        "stimulus_suspect": stim,
        "self_family_pairs": meta.get("self_family_pairs", []),
        "statistical_note": prim["aggregates"].get("statistical_note", ""),
        "corpus_summary": prim["aggregates"].get("corpus_summary", ""),
    }


def build_tidy_df(evaluation, stimulus_suspect):
    """Reusable helper: records -> tidy long DataFrame (one row per model×event×dim)."""
    df = pd.json_normalize(evaluation["records"])
    for col in ["bloc_signed_score", "signed_score", "raw_intensity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["self_family"] = df.get("self_family", pd.Series([False] * len(df))).fillna(False).astype(bool)
    df["stimulus_suspect"] = df["event_id"].isin(set(stimulus_suspect))
    all_flags = sorted({f for fl in df.get("flags", pd.Series([[]] * len(df))) for f in (fl or [])})
    for fg in all_flags:
        df[f"flag_{fg}"] = df["flags"].apply(lambda fl: fg in (fl or []))
    return df, all_flags


def grouped_mean_sd(df, cols, val="bloc_signed_score"):
    g = df.groupby(cols)[val].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def df_to_markdown(df):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + ["" if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ===========================================================================
# F1 -- Headline: model x dimension lean (comparison dims)
# ===========================================================================
def fig_f1(df, cfg):
    models = cfg["judged_models"]
    dims = cfg["comparison_dims"]
    clean = df[~df["stimulus_suspect"]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharey=True)
    axes = axes.reshape(-1)
    ymax = 0
    for ax, dim in zip(axes, dims):
        sub = clean[clean["dimension"] == dim]
        st = grouped_mean_sd(sub, ["model"]).set_index("model").reindex(models)
        for i, m in enumerate(models):
            mean, sd = st.loc[m, "mean"], st.loc[m, "std"]
            ax.bar(i, mean, yerr=sd, width=0.7, color=MODEL_COLORS[m],
                   edgecolor="white", linewidth=0.8,
                   error_kw=dict(ecolor="#9a9a9a", elinewidth=1.1, capsize=0),
                   hatch=SELF_HATCH if m == "claude" else None, zorder=3)
            ymax = max(ymax, abs(mean) + (0 if math.isnan(sd) else sd))
        ax.axhline(0, color="#333333", lw=1.0, zorder=2)
        ax.set_title(dim.replace("_", " "), fontsize=10.5, pad=6)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=8.5)
        ax.grid(axis="y")
        ax.tick_params(bottom=False)
    lim = 1.12 * ymax
    for ax in axes:
        ax.set_ylim(-lim, lim)
    fig.supylabel("mean bloc lean      − global south   ·   + western / NATO",
                  fontsize=9, color="#333333")

    titles(fig, "Where each model leans, per dimension",
           "Claude Opus 4.8 judge  ·  45 events (17 stimulus-suspect excluded, n=28/bar)  ·  "
           "bars = SD  ·  ╱╱ claude = self-scored")
    fig.tight_layout(rect=[0.02, 0, 1, 0.9])
    return _save(fig, "F1_model_x_dimension_lean.png")


# ===========================================================================
# F2 -- Lean by conflict (blame_attribution heatmap)
# ===========================================================================
def fig_f2(df, cfg):
    models = cfg["judged_models"]
    clean = df[~df["stimulus_suspect"]].copy()
    top = [c for c in ["ukraine_war", "lebanon_ceasefire", "gaza_ceasefire"]
           if c in clean["conflict_id"].unique()]
    clean["cg"] = clean["conflict_id"].where(clean["conflict_id"].isin(top), "other")
    conf = top + ["other"]
    sub = clean[clean["dimension"] == "blame_attribution"]

    mat = np.full((len(models), len(conf)), np.nan)
    ncol = []
    for j, c in enumerate(conf):
        ncol.append(sub[sub["cg"] == c]["event_id"].nunique())
        for i, m in enumerate(models):
            cell = sub[(sub["model"] == m) & (sub["cg"] == c)]
            if len(cell):
                mat[i, j] = cell["bloc_signed_score"].mean()

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(conf)))
    ax.set_xticklabels([f"{c.replace('_', ' ')}\n(n={n})" for c, n in zip(conf, ncol)], fontsize=9)
    ylab = [f"{m}  (self)" if m == "claude" else m for m in models]
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(ylab, fontsize=9)
    ax.tick_params(length=0)
    for i in range(len(models)):
        for j in range(len(conf)):
            if not math.isnan(mat[i, j]):
                lum = abs(mat[i, j]) / (vmax + 1e-9)
                ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center",
                        fontsize=10, color="white" if lum > 0.55 else "#222222")
    # one dashed outline around the self-scored claude row
    ci = models.index("claude")
    ax.add_patch(Rectangle((-0.5, ci - 0.5), len(conf), 1, fill=False,
                           edgecolor="#333333", lw=1.4, ls=(0, (3, 2)), zorder=5))
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("bloc lean  (− GS · + West)", fontsize=8.5)
    cb.outline.set_visible(False)

    titles(fig, "Blame-attribution lean by conflict",
           "Claude Opus 4.8 judge  ·  stimulus-suspect excluded  ·  cell = mean bloc lean",
           y=1.0, sub_dy=0.07)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _save(fig, "F2_lean_by_conflict.png")


# ===========================================================================
# F3 -- Inter-model disagreement ranking
# ===========================================================================
def fig_f3(opus, cfg):
    ima = sorted(opus["primary_evaluation"]["aggregates"]["inter_model_agreement"],
                 key=lambda x: x["mean_pairwise_diff"], reverse=True)
    dims = [d["dimension"].replace("_", " ") for d in ima]
    vals = [d["mean_pairwise_diff"] for d in ima]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(dims))[::-1]
    colors = [ACCENT if i == 0 else NEUTRAL for i in range(len(dims))]
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.8, height=0.72, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.03, yi, f"{v:.2f}", va="center", fontsize=9.5, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(dims, fontsize=9.5)
    ax.set_xlabel("mean pairwise | score difference |  between models")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="x")
    ax.tick_params(length=0)

    titles(fig, "Where the models diverge most",
           "blame attribution is the most-contested axis  ·  Claude Opus 4.8 judge",
           y=1.0, sub_dy=0.08)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    return _save(fig, "F3_inter_model_disagreement.png")


# ===========================================================================
# F4 -- Consistency: lean strength vs noise (all volatile)
# ===========================================================================
def fig_f4(opus, cfg):
    cons = opus["primary_evaluation"]["aggregates"]["consistency"]
    lim = max(max(c["mean_abs_lean"] for c in cons),
              max(c["mean_sd"] for c in cons)) * 1.18
    smax = max(c["n_sign_flips"] for c in cons) or 1

    fig, ax = plt.subplots(figsize=(7, 6))
    # shaded "noise > lean -> volatile" region
    ax.fill_between([0, lim], [0, lim], [lim, lim], color=ACCENT, alpha=0.05, zorder=0)
    ax.plot([0, lim], [0, lim], ls="--", color="#c9c9c9", lw=1.2, zorder=1)
    ax.text(0.30 * lim, 0.92 * lim, "noise > lean  →  volatile",
            fontsize=9, color="#b0592c", style="italic")
    lab_off = {"claude": (13, 7), "grok": (13, -4), "gpt-4.1": (-16, 15),
               "deepseek": (14, 6), "llama": (14, 4)}
    for c in cons:
        m = c["model"]
        size = 130 + 560 * (c["n_sign_flips"] / smax)
        ax.scatter(c["mean_abs_lean"], c["mean_sd"], s=size, color=MODEL_COLORS.get(m, "#888"),
                   edgecolor="white", linewidth=1.2, alpha=0.9, zorder=3,
                   hatch=SELF_HATCH if m == "claude" else None)
        ax.annotate(m, (c["mean_abs_lean"], c["mean_sd"]), textcoords="offset points",
                    xytext=lab_off.get(m, (12, 7)), fontsize=9, fontweight="bold", color="#333333")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("mean | lean |   (how strongly it leans)")
    ax.set_ylabel("mean SD   (how noisy across events)")
    ax.grid(True)
    ax.set_aspect("equal", "box")

    titles(fig, "Leanings exist, but none are stable",
           "all five models are 'volatile'  ·  marker size = sign flips  ·  ╱╱ claude self-scored",
           y=1.0, sub_dy=0.06)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _save(fig, "F4_consistency.png")


# ===========================================================================
# F6 -- Two-judge reliability (honest version)
# ===========================================================================
def fig_f6(opus, cfg):
    tj = opus["two_judge_reliability"]
    num, flag, frame = tj["numeric_score_agreement"], tj["flag_agreement"], tj["frame_agreement"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    # left: three headline numbers, numeric segregated as a validity check
    labels = ["frame\nclassification", "flag set\n(overall)", "numeric score\n(validity check)"]
    vals = [frame["classified_frame_agreement_pct"], flag["overall_pct"], num["exact_match_pct"]]
    cols = ["#009E73", "#E69F00", "#C6C6C6"]
    bars = ax1.bar(range(3), vals, color=cols, edgecolor="white", linewidth=0.9, width=0.7, zorder=3)
    bars[2].set_hatch("xx")
    for i, v in zip(range(3), vals):
        ax1.text(i, v + 1.6, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color="#333")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("agreement (%)")
    ax1.set_title("genuine agreement is moderate", fontsize=10.5, pad=6)
    ax1.grid(axis="y")
    ax1.tick_params(bottom=False)

    # right: flag agreement by dimension
    bydim = flag["by_dimension"]
    dims = [d for d in cfg["all_dims"] if d in bydim]
    dv = [bydim[d] for d in dims]
    y = np.arange(len(dims))[::-1]
    ax2.barh(y, dv, color="#E69F00", edgecolor="white", linewidth=0.8, height=0.72, zorder=3)
    ax2.axvline(flag["overall_pct"], ls="--", color=ACCENT, lw=1.3, zorder=4)
    ax2.text(flag["overall_pct"] + 1.5, -0.35, f"overall {flag['overall_pct']:.0f}%",
             fontsize=8, color=ACCENT, va="center", ha="left")
    for yi, v in zip(y, dv):
        ax2.text(v + 1.2, yi, f"{v:.0f}%", va="center", fontsize=9, color="#333")
    ax2.set_yticks(y)
    ax2.set_yticklabels([d.replace("_", " ") for d in dims], fontsize=9)
    ax2.set_xlim(0, 100)
    ax2.set_title("flag agreement by dimension", fontsize=10.5, pad=6)
    ax2.set_xlabel("exact flag-set match (%)")
    ax2.grid(axis="x")
    ax2.tick_params(length=0)

    dis = Counter(d["model"] for d in frame.get("disagreements", []))
    dis_txt = ", ".join(f"{m}" for m in dis) or "none"
    titles(fig, "Two-judge reliability (Opus vs. GPT-5.6, 21 shared events)",
           f"numeric match is deterministic, not judgment  ·  all {sum(dis.values())} frame "
           f"disagreements trace to {dis_txt}", y=1.0, sub_dy=0.065)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return _save(fig, "F6_two_judge_reliability.png")


# ===========================================================================
# F7 -- Mechanism: asymmetric credibility by pole-bloc
# ===========================================================================
def fig_f7(opus, cfg):
    ea = opus["primary_evaluation"]["event_axis"]
    stim = set(cfg["stimulus_suspect"])
    models = cfg["judged_models"]
    CRED = {"credible": 1.0, "unverified": 0.0, "unaddressed": 0.0, "propagandistic": -1.0}

    west, gs, hedge_total = defaultdict(list), defaultdict(list), 0
    for e in ea:
        if e["event_id"] in stim:
            continue
        pn, pp = e.get("pole_negative"), e.get("pole_positive")
        pnb, ppb = e.get("pole_negative_bloc"), e.get("pole_positive_bloc")
        for note in e.get("per_model_notes", []):
            m = note.get("model")
            hedge_total += sum(v for v in (note.get("hedge_counts") or {}).values()
                               if isinstance(v, (int, float)))
            for party, verdict in (note.get("credibility_verdict") or {}).items():
                if verdict not in CRED:
                    continue
                if party == pn:
                    (west if pnb == "western_nato_aligned" else gs)[m].append(CRED[verdict])
                elif party == pp:
                    (west if ppb == "western_nato_aligned" else gs)[m].append(CRED[verdict])

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(models))
    w = 0.38
    wv = [mean(west.get(m, [])) for m in models]
    gv = [mean(gs.get(m, [])) for m in models]
    ax.bar(x - w / 2, wv, w, color="#0072B2", edgecolor="white", linewidth=0.8,
           label="western / NATO-aligned claims", zorder=3)
    ax.bar(x + w / 2, gv, w, color="#D55E00", edgecolor="white", linewidth=0.8,
           label="global south / nonaligned claims", zorder=3)
    ax.axhline(0, color="#333333", lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(-0.75, 1.0)
    ax.set_ylabel("mean credibility   (+1 credible · −1 propagandistic)")
    ax.grid(axis="y")
    ax.tick_params(bottom=False)
    ax.legend(loc="upper left", ncol=1)

    titles(fig, "Which side's claims does a model credit?",
           "mechanism behind the lean  ·  credibility_verdict  ·  stimulus-suspect excluded",
           y=1.0, sub_dy=0.07)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return _save(fig, "F7_asymmetric_credibility.png")


# ===========================================================================
# stdout table + findings.md
# ===========================================================================
def summary_table(df, opus, cfg):
    models, comp = cfg["judged_models"], cfg["comparison_dims"]
    clean = df[~df["stimulus_suspect"]]
    rel = {c["model"]: c["reliability"] for c in opus["primary_evaluation"]["aggregates"]["consistency"]}
    rows = []
    for m in models:
        row = {"model": m}
        for d in comp:
            v = clean[(clean["model"] == m) & (clean["dimension"] == d)]["bloc_signed_score"].mean()
            row[d] = round(v, 2) if not math.isnan(v) else float("nan")
        row["reliability"] = rel.get(m, "?")
        row["self_scored"] = "yes" if clean[clean["model"] == m]["self_family"].any() else ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def write_findings(df, opus, cfg, table):
    tj = opus["two_judge_reliability"]
    ima = sorted(opus["primary_evaluation"]["aggregates"]["inter_model_agreement"],
                 key=lambda x: x["mean_pairwise_diff"], reverse=True)
    frame, flag, num = tj["frame_agreement"], tj["flag_agreement"], tj["numeric_score_agreement"]
    dis = Counter(d["model"] for d in frame.get("disagreements", []))
    top = ima[0]
    clean = df[~df["stimulus_suspect"]]
    mism = clean.groupby("model")["flag_label_lean_mismatch"].sum() \
        if "flag_label_lean_mismatch" in clean.columns else pd.Series(dtype=int)
    mism_str = ", ".join(f"{m}={int(mism.get(m, 0))}" for m in cfg["judged_models"])

    L = []
    A = L.append
    A("# Model narrative-bias study — findings\n")
    A("_Local analysis of `judge_opus_4.8.json` (primary, Claude Opus 4.8) and the embedded "
      "`second_judge_gpt_5_6` (GPT-5.6-sol). Descriptive, not inferential._\n")

    A("## Guardrails\n")
    A(f"- **Numeric-score agreement is not reliability.** `signed_score` is a deterministic "
      f"projection of (favored_party, raw_intensity, poles); the {num['exact_match_pct']:.0f}% "
      f"cross-judge match is an arithmetic **validity check**. Real reliability = flag agreement "
      f"(**{flag['overall_pct']:.1f}%**) and frame agreement (**{frame['classified_frame_agreement_pct']:.1f}%**).")
    A(f"- **stimulus_suspect events excluded from all lean claims** — {len(cfg['stimulus_suspect'])} "
      f"events: {cfg['stimulus_suspect']}.")
    A("- **Self-family cells flagged.** claude is scored by an Opus judge (Opus×claude, "
      "self_family=true); hatched in every figure showing claude's lean.")
    A("- **No model met the consistent-lean threshold** (|mean|>=1.5, sd<=1.5, >=70% same sign). "
      "All five are `volatile`: leanings present but volatile, not consistent.")
    A(f"- **n=45 events (21 two-judge).** SD error bars show spread, not significance.\n")
    A(f"> statistical_note: {cfg['statistical_note']}\n")

    A("## Figures (core set)\n")
    A("**F1 — Model × dimension lean.** Mean `bloc_signed_score` per model across the four "
      "comparison dimensions, stimulus_suspect excluded. Every SD bar dwarfs its mean — the visual "
      "signature of volatile, not consistent, leanings. grok is the lone slight western tilt; the "
      "others lean global-south. claude bars are hatched (self-scored).\n")
    A(f"**F2 — Lean by conflict.** Blame-attribution mean per model per conflict. Leans are "
      "conflict-specific: grok/gpt-4.1 tilt western on Ukraine but most models swing global-south on "
      "Lebanon/Gaza. Cell counts are small — read direction, not magnitude.\n")
    A(f"**F3 — Inter-model disagreement.** Dimensions ranked by mean pairwise |score diff|. "
      f"**{top['dimension']}** is the most-contested axis (**{top['mean_pairwise_diff']:.2f}**), "
      "driven by causal-start-point and actor-selection divergence.\n")
    A("**F4 — Consistency.** Lean strength vs. per-event noise, marker size = sign flips. Every "
      "model sits above the noise=lean line, so per-event noise ≈ or exceeds the average lean. All "
      "five are `volatile` — the 'leanings exist but aren't stable' finding.\n")
    A(f"**F6 — Two-judge reliability.** Frame agreement **{frame['classified_frame_agreement_pct']:.1f}%**, "
      f"flag agreement **{flag['overall_pct']:.1f}%**; the deterministic numeric match is shown only as "
      f"a separate validity-check bar. All {sum(dis.values())} frame disagreements trace to "
      f"{', '.join(dis) or 'none'} (non-standard declared labels).\n")
    A("**F7 — Asymmetric credibility (mechanism).** Mean credibility granted to western vs. "
      "global-south pole claims (credible=+1, propagandistic=−1). deepseek doubts western claims "
      "while crediting global-south — a concrete mechanism for its lean; grok is the only model "
      "crediting western more. (Intended `hedge_counts` is near-empty — 4 corpus-wide — so "
      "`credibility_verdict` is used instead.)\n")

    A("_Not shown (available on request): declared-vs-revealed frame mismatch "
      f"(label_lean_mismatch, stimulus-excluded: {mism_str}) and emotional-tone-vs-blame scatter._\n")

    A("## Per-model lean (comparison dimensions, stimulus_suspect excluded)\n")
    A(df_to_markdown(table) + "\n")

    A("## Most defensible findings\n")
    A(f"1. **Blame attribution is the most-contested dimension** (mean pairwise |diff| "
      f"{top['mean_pairwise_diff']:.2f}) — a clean, judge-independent structural result.")
    A("2. **Leanings are present but volatile.** No model met the consistent-lean threshold; all "
      "five are `volatile`, so bloc means are spreads, not point estimates.")
    A(f"3. **Judges agree on frames, only moderately on flags** "
      f"({frame['classified_frame_agreement_pct']:.1f}% vs. {flag['overall_pct']:.1f}%); the ~100% "
      "numeric match is a deterministic validity check. claude cells are self-scored under the Opus "
      "judge and carry that caveat.")

    out = os.path.join(HERE, "findings.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return out


def print_startup(opus, gpt_backup, cfg, df):
    prim = opus["primary_evaluation"]
    print("=" * 74)
    print("MODEL BIAS STUDY  —  local analysis  (no API, no network)")
    print("=" * 74)
    print("\nEVENT / RECORD COUNTS PER TIER")
    print(f"  primary_evaluation   : {len(prim['records']):>5} records, {len(prim['event_axis'])} events")
    sj = opus["second_judge_gpt_5_6"]
    print(f"  second_judge_gpt_5_6 : {len(sj['records']):>5} records, {len(sj['event_axis'])} events")
    print(f"  two_judge_reliability: {opus['two_judge_reliability']['n_shared_events']} shared events")
    print(f"  gpt backup file      : {len(gpt_backup['records'])} records (cross-check)")
    print(f"\nDIMENSIONS (from data; no 'geopolitical_perspective' present)")
    print(f"  comparison ({len(cfg['comparison_dims'])}): {cfg['comparison_dims']}")
    print(f"  mechanism  ({len(cfg['mechanism_dims'])}): {cfg['mechanism_dims']}")
    print(f"\nSTIMULUS_SUSPECT ({len(cfg['stimulus_suspect'])}) — excluded: {cfg['stimulus_suspect']}")
    print(f"\nSELF-FAMILY cells: {int(df['self_family'].sum())} records "
          f"({[(p['judge'].split('/')[-1], p['judged_model']) for p in cfg['self_family_pairs']]})")
    print("\nCONFLICT DISTRIBUTION (distinct events)")
    for c, n in df.groupby("conflict_id")["event_id"].nunique().sort_values(ascending=False).items():
        print(f"  {c:<22} {n:>3}")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", help="'f1' = startup + F1 only; 'all' = full set")
    args = ap.parse_args()
    apply_style()

    opus, gpt_backup, op, gp = load_data()
    print(f"[loaded] {op}\n[loaded] {gp}\n")
    cfg = get_config(opus)
    df, all_flags = build_tidy_df(opus["primary_evaluation"], cfg["stimulus_suspect"])
    print_startup(opus, gpt_backup, cfg, df)

    print(f"\n[F1] {fig_f1(df, cfg)}")
    if args.only.lower() == "f1":
        print("[checkpoint] F1 only; re-run with --only all for the full set.")
        return

    for fn, tag in [(lambda: fig_f2(df, cfg), "F2"), (lambda: fig_f3(opus, cfg), "F3"),
                    (lambda: fig_f4(opus, cfg), "F4"), (lambda: fig_f6(opus, cfg), "F6"),
                    (lambda: fig_f7(opus, cfg), "F7")]:
        print(f"[{tag}] {fn()}")

    table = summary_table(df, opus, cfg)
    print("\n" + "=" * 74)
    print("PER-MODEL LEAN (comparison dimensions, stimulus_suspect excluded)")
    print("=" * 74)
    with pd.option_context("display.width", 120):
        print(table.to_string())
    print("=" * 74)
    print(f"\n[findings] {write_findings(df, opus, cfg, table)}")
    print(f"[done] core figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
