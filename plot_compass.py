#!/usr/bin/env python3
"""
plot_compass.py -- "political compass" + radar graphics for the model-lean study.

Pure local analysis (pandas + matplotlib, no seaborn, no network, no API).

Places the 5 judged models (claude, deepseek, grok, gpt-4.1, llama) in
political-lean space using the Opus-4.8 primary_evaluation block.

Sign / axis convention (stated on every figure):
    X-axis = bloc lean.
        LEFT  = global_south / nonaligned   (negative bloc_signed_score)
        RIGHT = western_nato_aligned         (positive bloc_signed_score)
        0     = neutral center.

Guardrails (match the study's honesty rules):
    * stimulus_suspect events are EXCLUDED from all lean math (flagged in data,
      read from event_axis, not hardcoded).
    * claude cells are self-scored by the Opus judge -> flagged in both figures.
    * SD whiskers / spread shown; NO significance stars (n small, descriptive only).
    * These are RELATIVE leanings on contested events with no neutral truth.
      A near-0 model "does not lean" -- it is NOT "unbiased / correct".

Usage:
    python plot_compass.py            # table + both figures + csv
    python plot_compass.py compass    # table + compass only
    python plot_compass.py radar      # table + radar only
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_PATH = Path("final_judge/judge_opus_4.8.json")
FIG_DIR = Path("figures")
CSV_PATH = Path("compass_data.csv")

# Fixed model order (worst-west -> most global-south, per sanity expectation)
MODEL_ORDER = ["grok", "claude", "gpt-4.1", "llama", "deepseek"]

# The 4 signed COMPARISON dimensions used on the radar.
# (geopolitical_perspective is a frame, not a signed score -> omitted.)
COMPARISON_DIMS = [
    "blame_attribution",
    "moral_framing",
    "perceived_legitimacy",
    "emotional_tone",
]
HEADLINE_DIM = "blame_attribution"
SELF_SCORED_MODEL = "claude"  # self_family cells -> contamination-prone

# Colorblind-safe (Okabe-Ito) per-model color map -- defined ONCE, reused in both figs.
MODEL_COLORS = {
    "grok":     "#D55E00",  # vermillion
    "claude":   "#0072B2",  # blue
    "gpt-4.1":  "#009E73",  # bluish green
    "llama":    "#CC79A7",  # reddish purple
    "deepseek": "#E69F00",  # orange
}

# Diverging half-tints for the compass background (colorblind-safe, very light).
LEFT_TINT = "#56B4E9"   # sky blue  -> global-south side
RIGHT_TINT = "#E69F00"  # orange    -> western side

SIGN_LINE = "+X = western / -X = global-south  (0 = neutral center)"


# --------------------------------------------------------------------------- #
# Data loading & lean math
# --------------------------------------------------------------------------- #
def load_primary(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    return blob["primary_evaluation"]


def suspect_event_ids(event_axis):
    """Read stimulus_suspect events from the data -- do NOT hardcode."""
    return {
        e["event_id"]
        for e in event_axis
        if e.get("stimulus_check", {}).get("stimulus_suspect") is True
    }


def build_frame(primary):
    """Records -> tidy DataFrame with a clean-event mask."""
    df = pd.DataFrame(primary["records"])
    suspect = suspect_event_ids(primary["event_axis"])
    df["stimulus_suspect"] = df["event_id"].isin(suspect)
    df["clean"] = ~df["stimulus_suspect"]
    return df, suspect


def per_event_leans(df, model, dim):
    """
    Mean bloc_signed_score per event for one model+dimension on CLEAN events.
    One value per event (records are already 1-per model/event/dim, but we
    group defensively in case of duplicates).
    """
    sub = df[(df["model"] == model) & (df["dimension"] == dim) & df["clean"]]
    return sub.groupby("event_id")["bloc_signed_score"].mean()


def compute_metrics(df):
    """
    Per model:
        X   = mean bloc_signed_score on blame_attribution (clean)
        Y   = mean of |per-event bloc lean| on blame_attribution (clean)
              -- lean INTENSITY: average of absolute per-event leans, NOT |mean|,
                 so a model that swings hard both ways still scores high.
        SD  = per-event SD of blame lean (spread)
        + mean bloc_signed_score for each of the 4 comparison dims (radar).
    """
    rows = []
    for m in MODEL_ORDER:
        blame = per_event_leans(df, m, HEADLINE_DIM)
        rec = {
            "model": m,
            "n_events": int(blame.shape[0]),
            "x_blame_mean": float(blame.mean()),
            "y_intensity": float(blame.abs().mean()),
            "sd_blame": float(blame.std(ddof=0)),  # population SD across events
            "self_scored": (m == SELF_SCORED_MODEL),
        }
        for dim in COMPARISON_DIMS:
            rec[dim] = float(per_event_leans(df, m, dim).mean())
        rows.append(rec)
    return pd.DataFrame(rows).set_index("model").loc[MODEL_ORDER]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_table(metrics, n_clean, n_suspect):
    print()
    print("=" * 78)
    print("  MODEL LEAN METRICS  (Opus-4.8 judge, primary_evaluation)")
    print(f"  clean events = {n_clean}   |   stimulus_suspect events EXCLUDED = {n_suspect}")
    print("  X = blame mean (+west / -global-south) ; Y = |per-event| intensity")
    print("=" * 78)
    hdr = (f"{'model':<9} {'X_blame':>8} {'Y_int':>7} {'SD':>6}  |  "
           f"{'blame':>7} {'moral':>7} {'legit':>7} {'emot':>7}  self?")
    print(hdr)
    print("-" * 78)
    for m in MODEL_ORDER:
        r = metrics.loc[m]
        flag = "SELF" if r["self_scored"] else ""
        print(f"{m:<9} {r['x_blame_mean']:>+8.2f} {r['y_intensity']:>7.2f} "
              f"{r['sd_blame']:>6.2f}  |  "
              f"{r['blame_attribution']:>+7.2f} {r['moral_framing']:>+7.2f} "
              f"{r['perceived_legitimacy']:>+7.2f} {r['emotional_tone']:>+7.2f}  {flag}")
    print("-" * 78)
    # Sanity confirmation
    xs = metrics["x_blame_mean"]
    grok_pos = xs.loc["grok"] > 0
    ds_min = xs.idxmin() == "deepseek"
    print(f"  sanity: grok positive (western)?  {grok_pos}   "
          f"deepseek most negative?  {ds_min}")
    print("=" * 78)
    print()


def write_csv(metrics, path=CSV_PATH):
    out = metrics.reset_index()[
        ["model", "n_events", "x_blame_mean", "y_intensity", "sd_blame",
         "blame_attribution", "moral_framing", "perceived_legitimacy",
         "emotional_tone", "self_scored"]
    ]
    out.to_csv(path, index=False)
    print(f"[wrote] {path}")


# --------------------------------------------------------------------------- #
# GRAPHIC 1 -- the compass
# --------------------------------------------------------------------------- #
def plot_compass(metrics, n_clean, stat_note):
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 8.5))

    xs = metrics["x_blame_mean"]
    ys = metrics["y_intensity"]

    # Symmetric X range around 0 with headroom.
    xabs = max(3.2, float(xs.abs().max()) + 0.9)
    xlo, xhi = -xabs, xabs
    ylo = 0.0
    yhi = max(4.2, float(ys.max()) + 0.8)
    y_mid = float(ys.mean())  # mid-intensity divider

    # Quadrant tints: left = global-south, right = western (very light).
    ax.axvspan(xlo, 0, color=LEFT_TINT, alpha=0.10, zorder=0)
    ax.axvspan(0, xhi, color=RIGHT_TINT, alpha=0.10, zorder=0)

    # Divider lines.
    ax.axvline(0, color="#444444", lw=1.4, zorder=1)
    ax.axhline(y_mid, color="#999999", lw=1.0, ls="--", zorder=1)
    ax.text(xhi * 0.995, y_mid, "  mid intensity", va="bottom", ha="right",
            fontsize=8, color="#777777", style="italic")

    # Corner side labels.
    ax.text(xlo * 0.97, yhi * 0.97, "Global-South leaning", ha="left", va="top",
            fontsize=13, fontweight="bold", color="#1f6699")
    ax.text(xhi * 0.97, yhi * 0.97, "Western leaning", ha="right", va="top",
            fontsize=13, fontweight="bold", color="#9c6b00")

    # Points + SD whiskers on X.
    for m in MODEL_ORDER:
        r = metrics.loc[m]
        x, y, sd = r["x_blame_mean"], r["y_intensity"], r["sd_blame"]
        color = MODEL_COLORS[m]
        self_scored = r["self_scored"]

        # per-event SD whisker on X
        ax.errorbar(x, y, xerr=sd, fmt="none", ecolor=color, elinewidth=1.6,
                    capsize=5, capthick=1.6, alpha=0.75, zorder=3)

        if self_scored:
            # ring-outline + hatch to flag contamination-prone self-scored cell
            ax.scatter([x], [y], s=340, facecolor=color, edgecolor="black",
                       linewidth=2.2, hatch="////", zorder=5)
            ax.scatter([x], [y], s=620, facecolor="none", edgecolor="black",
                       linewidth=1.4, linestyle=(0, (2, 2)), zorder=5)
        else:
            ax.scatter([x], [y], s=300, facecolor=color, edgecolor="black",
                       linewidth=1.2, zorder=5)

        label = m + ("\n(self-scored by Opus judge)" if self_scored else "")
        dy = 0.16 if y < yhi - 0.6 else -0.30
        va = "bottom" if dy > 0 else "top"
        ax.annotate(label, (x, y), xytext=(0, 12 if dy > 0 else -12),
                    textcoords="offset points", ha="center", va=va,
                    fontsize=10.5, fontweight="bold", color="black",
                    zorder=6)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("bloc lean on blame attribution   "
                  "(-  global-south / nonaligned      western / NATO-aligned  +)",
                  fontsize=11)
    ax.set_ylabel("lean intensity  (mild → strong)", fontsize=11)
    ax.set_title("Where five models land on blame attribution "
                 "(neutral-tone contested events)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.15, zorder=0)

    caption = (
        f"X = mean bloc_signed_score on blame_attribution over n={n_clean} clean events "
        f"(stimulus_suspect events excluded).  {SIGN_LINE}\n"
        "Y = mean of |per-event bloc lean| (per-event conviction the mean can hide); "
        "X & Y are related but a model that swings hard both ways still scores high on Y. "
        "Whiskers = per-event SD.\n"
        "Descriptive, not inferential: relative leanings on contested events with no neutral "
        "truth. A near-0 model \"does not lean\" — not \"unbiased/correct\". No significance "
        "testing. claude cells self-scored by the Opus judge.\n"
        f"statistical_note: {stat_note}"
    )
    fig.text(0.065, -0.02, caption, ha="left", va="top", fontsize=8.0,
             color="#333333", wrap=True)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = FIG_DIR / "compass_blame.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")
    return out


# --------------------------------------------------------------------------- #
# GRAPHIC 2 -- the radar / spider
# --------------------------------------------------------------------------- #
def plot_radar(metrics, n_clean, stat_note):
    FIG_DIR.mkdir(exist_ok=True)

    dims = COMPARISON_DIMS
    nice = {
        "blame_attribution": "blame\nattribution",
        "moral_framing": "moral\nframing",
        "perceived_legitimacy": "perceived\nlegitimacy",
        "emotional_tone": "emotional\ntone",
    }
    N = len(dims)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles_closed = angles + [angles[0]]

    # Signed -5..+5 -> radial 0..10 with neutral ring (0) at radius 5.
    RMAX = 5.0
    def to_r(v):
        return float(np.clip(v, -RMAX, RMAX) + RMAX)

    fig = plt.figure(figsize=(10, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels([nice[d] for d in dims], fontsize=11, fontweight="bold")

    ax.set_ylim(0, 2 * RMAX)
    rticks = [0, 2.5, 5, 7.5, 10]
    ax.set_yticks(rticks)
    ax.set_yticklabels(["-5", "-2.5", "0", "+2.5", "+5"], fontsize=8, color="#666666")
    ax.set_rlabel_position(90)

    # Neutral ring at radius 5 (bloc_signed_score = 0), drawn & labelled explicitly.
    ax.plot(np.linspace(0, 2 * np.pi, 200), [5] * 200, color="#444444",
            lw=1.6, ls="--", zorder=2)
    ax.text(np.deg2rad(90), 5.0, "  neutral (0)", color="#444444", fontsize=9,
            ha="left", va="center", fontstyle="italic", zorder=6)

    # Tint inside neutral ring = global-south, outside = western.
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill_between(theta, 0, 5, color=LEFT_TINT, alpha=0.06, zorder=0)
    ax.fill_between(theta, 5, 10, color=RIGHT_TINT, alpha=0.06, zorder=0)

    for m in MODEL_ORDER:
        r = metrics.loc[m]
        vals = [to_r(r[d]) for d in dims]
        vals_closed = vals + [vals[0]]
        color = MODEL_COLORS[m]
        self_scored = r["self_scored"]
        ls = (0, (4, 2)) if self_scored else "-"
        lw = 2.0 if self_scored else 2.2
        ax.plot(angles_closed, vals_closed, color=color, lw=lw, ls=ls, zorder=4)
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.08, zorder=3)
        ax.scatter(angles, vals, color=color, s=28, edgecolor="black",
                   linewidth=0.6, zorder=5)

    ax.set_title("Per-dimension lean profile by model", fontsize=14,
                 fontweight="bold", pad=26)

    # Legend (per-model colors reused from compass; claude flagged self-scored).
    handles = []
    for m in MODEL_ORDER:
        lbl = m + ("  (self-scored)" if m == SELF_SCORED_MODEL else "")
        ls = (0, (4, 2)) if m == SELF_SCORED_MODEL else "-"
        handles.append(Line2D([0], [0], color=MODEL_COLORS[m], lw=2.4, ls=ls, label=lbl))
    handles.append(Patch(facecolor=LEFT_TINT, alpha=0.25,
                         label="inward = global-south"))
    handles.append(Patch(facecolor=RIGHT_TINT, alpha=0.25,
                         label="outward = western"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.32, 1.10),
              fontsize=9, frameon=True)

    caption = (
        f"Radial value = mean bloc_signed_score per dimension over n={n_clean} clean events "
        f"(stimulus_suspect excluded).  {SIGN_LINE}\n"
        "Dashed neutral ring = 0; inward of it = global-south lean, outward = western lean. "
        "Polygons keep roughly the SAME shape across dimensions — the leans are correlated, "
        "which is the honest reason the compass uses one dimension, not four fake axes.\n"
        "Descriptive, not inferential: relative leanings on contested events with no neutral "
        "truth; a near-neutral model \"does not lean\", not \"unbiased/correct\". No significance "
        "testing. claude polygon (dashed) is self-scored by the Opus judge.\n"
        f"statistical_note: {stat_note}"
    )
    fig.text(0.06, 0.02, caption, ha="left", va="bottom", fontsize=8.0,
             color="#333333", wrap=True)

    fig.tight_layout(rect=(0, 0.10, 0.98, 1))
    out = FIG_DIR / "radar_dimensions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    primary = load_primary()
    df, suspect = build_frame(primary)
    n_suspect = len(suspect)
    n_clean = int(df[df["dimension"] == HEADLINE_DIM]["clean"].sum() /
                  df["model"].nunique())  # clean events per model
    stat_note = primary["aggregates"].get("statistical_note", "(none)")

    metrics = compute_metrics(df)
    print_table(metrics, n_clean, n_suspect)
    write_csv(metrics)

    if which in ("all", "compass"):
        plot_compass(metrics, n_clean, stat_note)
    if which in ("all", "radar"):
        plot_radar(metrics, n_clean, stat_note)


if __name__ == "__main__":
    main()
