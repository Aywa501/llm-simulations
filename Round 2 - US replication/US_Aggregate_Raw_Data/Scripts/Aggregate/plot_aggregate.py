"""
plot_aggregate.py

Generates two figures from aggregate_comparison_table_{cfg}.csv:

  1. aggregate_scatter_{cfg}.pdf
     LLM mean vs human mean, one point per arm×outcome.
     Points coloured by study; diagonal = perfect prediction; r annotated.

  2. aggregate_per_study_bars_{cfg}.pdf
     One panel per study; grouped bar chart (human vs LLM) per arm.
     Incomparable rows hatched and labelled.

Usage:
    python plot_aggregate.py [--config no_reasoning]
"""

import argparse, csv, textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parents[1] / "Data"
FIGS_DIR   = SCRIPT_DIR.parents[1] / "Figures"

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="no_reasoning")
args = parser.parse_args()

CSV_PATH    = DATA_DIR / f"aggregate_comparison_table_{args.config}.csv"
SCATTER_OUT = FIGS_DIR / f"aggregate_scatter_{args.config}.pdf"
BARS_OUT    = FIGS_DIR / f"aggregate_per_study_bars_{args.config}.pdf"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette — generate dynamically for however many studies appear
# ---------------------------------------------------------------------------

_BASE_COLOURS = [
    "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#F44336", "#795548", "#00BCD4", "#E91E63",
    "#3F51B5", "#009688", "#FF5722", "#607D8B",
    "#8BC34A", "#FFC107", "#673AB7", "#03A9F4",
]

def build_palette(seq_ids: list[int]) -> dict[int, str]:
    return {sid: _BASE_COLOURS[i % len(_BASE_COLOURS)]
            for i, sid in enumerate(sorted(seq_ids))}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_rows() -> list[dict]:
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            r["seq_id"]     = int(r["seq_id"])
            r["comparable"] = r["comparable"] == "True"
            for k in ("llm_mean", "human_mean"):
                r[k] = float(r[k]) if r[k] not in ("", "None") else None
            for k in ("llm_n", "human_n"):
                r[k] = int(r[k]) if r[k] not in ("", "None") else None
            rows.append(r)
    return rows

# ---------------------------------------------------------------------------
# Figure 1 — scatter
# ---------------------------------------------------------------------------

def plot_scatter(rows: list[dict], palette: dict):
    comp = [r for r in rows
            if r["comparable"]
            and r["human_mean"] is not None
            and r["llm_mean"]   is not None]

    if not comp:
        print("No comparable rows — skipping scatter.")
        return

    human = np.array([r["human_mean"] for r in comp])
    llm   = np.array([r["llm_mean"]   for r in comp])
    r_val, p_val = stats.pearsonr(human, llm)

    fig, ax = plt.subplots(figsize=(6, 6))

    lo = min(human.min(), llm.min()) * 0.95
    hi = max(human.max(), llm.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4, label="Perfect prediction")

    seen: set[int] = set()
    for r in comp:
        sid = r["seq_id"]
        col = palette[sid]
        # Shorten label for legend
        short_label = textwrap.shorten(r["study_label"], 35)
        lab = f"seq={sid}: {short_label}" if sid not in seen else None
        seen.add(sid)
        ax.scatter(r["human_mean"], r["llm_mean"],
                   c=col, s=60, alpha=0.85, edgecolors="white", lw=0.3,
                   label=lab, zorder=3)

    # OLS fit
    m, b = np.polyfit(human, llm, 1)
    x_fit = np.linspace(lo, hi, 200)
    ax.plot(x_fit, m * x_fit + b, color="#333", lw=1.2, alpha=0.6,
            label=f"OLS fit (r={r_val:.2f}, p={p_val:.3f})")

    ax.set_xlabel("Human mean", fontsize=12)
    ax.set_ylabel("LLM mean",   fontsize=12)
    ax.set_title(f"LLM vs Human arm means — aggregate studies\n"
                 f"(comparable arms only, config={args.config})", fontsize=11)
    ax.legend(fontsize=7, framealpha=0.9, loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(SCATTER_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {SCATTER_OUT}")
    print(f"  Pearson r={r_val:.3f}  p={p_val:.3f}  n={len(comp)} arms")

# ---------------------------------------------------------------------------
# Figure 2 — per-study bar charts
# ---------------------------------------------------------------------------

def short_label(s: str, max_len: int = 18) -> str:
    return textwrap.shorten(s.replace("_", " "), max_len)


def plot_per_study(rows: list[dict], palette: dict):
    by_study: dict[int, list] = defaultdict(list)
    for r in rows:
        by_study[r["seq_id"]].append(r)

    seq_ids   = sorted(by_study)
    n_studies = len(seq_ids)
    if n_studies == 0:
        print("No studies — skipping per-study bars.")
        return

    ncols = min(3, n_studies)
    nrows = (n_studies + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten() if n_studies > 1 else np.array([axes])

    for ax, seq_id in zip(axes, seq_ids):
        study_rows = sorted(by_study[seq_id], key=lambda r: r["arm_id"])
        arms       = [r["arm_id"]    for r in study_rows]
        short_arms = [short_label(a) for a in arms]
        human      = [r["human_mean"] for r in study_rows]
        llm        = [r["llm_mean"]   for r in study_rows]
        comp       = [r["comparable"] for r in study_rows]
        col        = palette[seq_id]

        x     = np.arange(len(arms))
        width = 0.35

        ax.bar(x - width / 2,
               [v if v is not None else 0 for v in human],
               width, label="Human", color="#BDBDBD", zorder=2)

        for i, (val, c) in enumerate(zip(llm, comp)):
            ax.bar(x[i] + width / 2, val if val is not None else 0,
                   width,
                   color=col, alpha=0.85,
                   hatch="" if c else "///",
                   edgecolor="white" if c else col,
                   zorder=2)

        title = textwrap.shorten(
            f"seq={seq_id} — {study_rows[0]['study_label']}", 55
        )
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(short_arms, rotation=35, ha="right", fontsize=6)
        ax.set_ylabel("Mean", fontsize=7)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)

        if seq_id == seq_ids[0]:
            ax.legend(
                handles=[
                    mpatches.Patch(color="#BDBDBD", label="Human"),
                    mpatches.Patch(color=col, label="LLM"),
                    mpatches.Patch(color=col, hatch="///",
                                   label="LLM (not comparable)"),
                ],
                fontsize=6, loc="upper right",
            )

    for ax in axes[n_studies:]:
        ax.set_visible(False)

    fig.suptitle(f"Human vs LLM arm means — aggregate studies ({args.config})",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(BARS_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {BARS_OUT}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows    = load_rows()
    seq_ids = sorted({r["seq_id"] for r in rows})
    palette = build_palette(seq_ids)
    print(f"Loaded {len(rows)} rows from {CSV_PATH.name}  ({len(seq_ids)} studies)")
    plot_scatter(rows, palette)
    plot_per_study(rows, palette)
