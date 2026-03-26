"""
plot_results.py

Generates two figures from comparison_table.csv:

  1. scatter_llm_vs_human.pdf
     Main replication scatter: LLM mean vs human mean, one point per arm×outcome.
     Points coloured by study; diagonal = perfect prediction; r annotated.

  2. per_study_bars.pdf
     One panel per study; grouped bar chart (human vs LLM) per arm.
     Incomparable rows shown with hatching and labelled.

Usage:
    python plot_results.py
"""

import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

DATA_DIR  = Path(__file__).resolve().parents[2] / "Data" / "Microdata"
FIGS_DIR  = Path(__file__).resolve().parents[2] / "Figures"
CSV_PATH  = DATA_DIR / "comparison_table.csv"

FIGS_DIR.mkdir(parents=True, exist_ok=True)

STUDY_LABELS = {
    6:  "Kidney allocation",
    20: "Xenophobia beliefs",
    28: "Islamophobia beliefs",
    29: "Legitimacy (dictator)",
    46: "Hiring discrimination",
    52: "Job attraction (demog.)",
    53: "Job attraction (bonus)",
}

PALETTE = {
    6:  "#2196F3",
    20: "#4CAF50",
    28: "#FF9800",
    29: "#9C27B0",
    46: "#F44336",
    52: "#795548",
    53: "#00BCD4",
}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_rows():
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

def plot_scatter(rows):
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

    # diagonal
    lo, hi = min(human.min(), llm.min()) * 0.95, max(human.max(), llm.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4, label="Perfect prediction")

    # scatter by study
    seen = set()
    for r in comp:
        sid  = r["seq_id"]
        col  = PALETTE[sid]
        lab  = STUDY_LABELS[sid] if sid not in seen else None
        seen.add(sid)
        ax.scatter(r["human_mean"], r["llm_mean"],
                   c=col, s=60, alpha=0.85, edgecolors="white", lw=0.3,
                   label=lab, zorder=3)

    # OLS fit line
    m, b = np.polyfit(human, llm, 1)
    x_fit = np.linspace(lo, hi, 200)
    ax.plot(x_fit, m * x_fit + b, color="#333", lw=1.2, alpha=0.6,
            label=f"OLS fit (r={r_val:.2f}, p={p_val:.3f})")

    ax.set_xlabel("Human mean", fontsize=12)
    ax.set_ylabel("LLM mean",   fontsize=12)
    ax.set_title("LLM vs Human arm means\n(comparable arms only)", fontsize=13)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    out = FIGS_DIR / "scatter_llm_vs_human.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
    print(f"  Pearson r={r_val:.3f}  p={p_val:.3f}  n={len(comp)} arms")


# ---------------------------------------------------------------------------
# Figure 2 — per-study bar charts
# ---------------------------------------------------------------------------

def short_arm(arm_id, max_len=18):
    return textwrap.shorten(arm_id.replace("_", " "), max_len)


def plot_per_study(rows):
    by_study = defaultdict(list)
    for r in rows:
        by_study[r["seq_id"]].append(r)

    seq_ids = sorted(by_study)
    n_studies = len(seq_ids)
    ncols = 3
    nrows = (n_studies + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, seq_id in zip(axes, seq_ids):
        study_rows = sorted(by_study[seq_id], key=lambda r: r["sim_arm_id"])
        arms = [r["sim_arm_id"] for r in study_rows]
        short_arms = [short_arm(a) for a in arms]
        human = [r["human_mean"] for r in study_rows]
        llm   = [r["llm_mean"]   for r in study_rows]
        comp  = [r["comparable"] for r in study_rows]

        x     = np.arange(len(arms))
        width = 0.35
        col   = PALETTE[seq_id]

        # human bars
        h_bars = ax.bar(x - width / 2,
                        [v if v is not None else 0 for v in human],
                        width, label="Human", color="#BDBDBD", zorder=2)

        # llm bars — hatched if not comparable
        for i, (val, c) in enumerate(zip(llm, comp)):
            ax.bar(x[i] + width / 2, val if val is not None else 0,
                   width,
                   color=col, alpha=0.85,
                   hatch="" if c else "///",
                   edgecolor="white" if c else col,
                   zorder=2)

        ax.set_title(f"seq={seq_id} — {STUDY_LABELS[seq_id]}", fontsize=9, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(short_arms, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Mean", fontsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
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
                fontsize=7, loc="upper right",
            )

    # hide unused axes
    for ax in axes[n_studies:]:
        ax.set_visible(False)

    fig.suptitle("Human vs LLM arm means by study", fontsize=14, y=1.01)
    out = FIGS_DIR / "per_study_bars.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = load_rows()
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")
    plot_scatter(rows)
    plot_per_study(rows)
