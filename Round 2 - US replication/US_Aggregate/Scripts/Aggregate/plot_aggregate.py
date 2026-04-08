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

CSV_PATH          = DATA_DIR / "Results" / f"aggregate_comparison_table_{args.config}.csv"
SCATTER_OUT       = FIGS_DIR / f"aggregate_scatter_{args.config}.pdf"
BARS_OUT          = FIGS_DIR / f"aggregate_per_study_bars_{args.config}.pdf"
COMBINED_OUT      = FIGS_DIR / f"aggregate_combined_{args.config}.pdf"
COMBINED_FILT_OUT = FIGS_DIR / f"aggregate_combined_no_collapsed_{args.config}.pdf"
COMBINED_SOUND_OUT = FIGS_DIR / f"aggregate_combined_sound_{args.config}.pdf"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Studies with verified data-quality problems that make simulation results
# uninterpretable:
#   151 — matrix-puzzle outcomes are a visual performance task (LLM can't see image)
#   164 — outcomes require clicking/listening to audio recordings
#   169 — arms and outcomes are OK but it's a real-money dictator game whose
#          binary choices collapse to YES/YES; included here for conservatism
#   174 — question_text = None for all 15 outcomes; LLM doesn't know what it's
#          answering YES/NO to
#   176 — all arm texts are identical (same political-identity prompt for every arm)
UNSOUND_STUDIES: set[int] = {151, 164, 169, 174, 176}

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
            for k in ("llm_mean", "human_mean", "human_norm", "llm_norm"):
                r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
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
# Figure 3 — combined: normalized scatter (top) + per-study r bars (bottom)
# ---------------------------------------------------------------------------

def _compute_norm_comp(comp: list[dict]) -> list[dict]:
    """Return a copy of comp rows augmented with within-study z-scored human_norm / llm_norm.
    Computed from human_mean / llm_mean so it is always correct for the data actually passed in
    (works whether rows come from the full dataset or a filtered subset).
    """
    by_study_idx: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(comp):
        by_study_idx[r["seq_id"]].append(i)

    augmented = [dict(r) for r in comp]   # shallow copies

    for seq_id, indices in by_study_idx.items():
        h = np.array([comp[i]["human_mean"] for i in indices], dtype=float)
        l = np.array([comp[i]["llm_mean"]   for i in indices], dtype=float)
        if len(h) >= 2 and h.std() > 0 and l.std() > 0:
            h_z = (h - h.mean()) / h.std()
            l_z = (l - l.mean()) / l.std()
            for j, i in enumerate(indices):
                augmented[i]["human_norm"] = float(h_z[j])
                augmented[i]["llm_norm"]   = float(l_z[j])
        else:
            for i in indices:
                augmented[i]["human_norm"] = None
                augmented[i]["llm_norm"]   = None

    return augmented


def plot_combined(rows: list[dict], palette: dict,
                  out_path: Path | None = None,
                  title_suffix: str = ""):
    """Single PDF combining a within-study normalized scatter and per-study r bars.

    Parameters
    ----------
    rows         : all rows from the CSV (comparable + non-comparable)
    palette      : seq_id → colour mapping
    out_path     : output PDF path; defaults to COMBINED_OUT
    title_suffix : extra string appended to the figure suptitle
    """
    out_path = out_path or COMBINED_OUT

    comp = [r for r in rows
            if r["comparable"]
            and r["human_mean"] is not None
            and r["llm_mean"]   is not None]

    if not comp:
        print(f"No comparable rows — skipping {out_path.name}.")
        return

    # Within-study z-scores computed fresh from human_mean / llm_mean
    comp_aug = _compute_norm_comp(comp)
    norm_comp = [r for r in comp_aug
                 if r.get("human_norm") is not None and r.get("llm_norm") is not None]

    # Build per-study data
    by_study: dict[int, list] = defaultdict(list)
    for r in comp:
        by_study[r["seq_id"]].append(r)

    all_seq_ids    = sorted({r["seq_id"] for r in rows})
    study_label_map: dict[int, str] = {}
    for r in rows:
        study_label_map[r["seq_id"]] = r["study_label"]

    # Per-study r and n
    study_r: dict[int, float | None] = {}
    study_n: dict[int, int]          = {}
    per_study_r_n: list[tuple[float, int]] = []

    for seq_id in all_seq_ids:
        study_rows = by_study.get(seq_id, [])
        study_n[seq_id] = len(study_rows)
        if len(study_rows) >= 2:
            h = np.array([r["human_mean"] for r in study_rows], dtype=float)
            l = np.array([r["llm_mean"]   for r in study_rows], dtype=float)
            if h.std() > 0 and l.std() > 0:
                rr, _ = stats.pearsonr(h, l)
                study_r[seq_id] = float(rr)
                per_study_r_n.append((float(rr), len(study_rows)))
            else:
                study_r[seq_id] = None
        else:
            study_r[seq_id] = None

    # ---- Layout ------------------------------------------------------------
    fig = plt.figure(figsize=(13, 11))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.85], hspace=0.45)
    ax_sc   = fig.add_subplot(gs[0])
    ax_bars = fig.add_subplot(gs[1])

    # === Top: normalized scatter ============================================
    if norm_comp:
        h_n = np.array([r["human_norm"] for r in norm_comp])
        l_n = np.array([r["llm_norm"]   for r in norm_comp])
        r_norm, p_norm = stats.pearsonr(h_n, l_n)

        pad = 0.4
        lo = min(h_n.min(), l_n.min()) - pad
        hi = max(h_n.max(), l_n.max()) + pad
        ax_sc.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.35,
                   label="Perfect prediction")
        ax_sc.axhline(0, color="gray", lw=0.4, alpha=0.4)
        ax_sc.axvline(0, color="gray", lw=0.4, alpha=0.4)

        seen: set[int] = set()
        for r in norm_comp:
            sid = r["seq_id"]
            col = palette[sid]
            short = textwrap.shorten(r["study_label"], 32)
            lab   = f"seq={sid}: {short}" if sid not in seen else None
            seen.add(sid)
            ax_sc.scatter(r["human_norm"], r["llm_norm"],
                          c=col, s=50, alpha=0.85, edgecolors="white", lw=0.3,
                          label=lab, zorder=3)

        m, b = np.polyfit(h_n, l_n, 1)
        x_fit = np.linspace(lo, hi, 200)
        ax_sc.plot(x_fit, m * x_fit + b, color="#333", lw=1.2, alpha=0.6,
                   label=f"OLS fit  r={r_norm:.2f}, p={p_norm:.3f}")

        ax_sc.set_xlabel("Human mean  (within-study z-score)", fontsize=11)
        ax_sc.set_ylabel("LLM mean  (within-study z-score)",   fontsize=11)
        ax_sc.set_title(
            f"Normalized LLM vs Human arm means — {args.config}\n"
            f"Within-study z-scoring removes cross-study scale differences",
            fontsize=10,
        )
        ax_sc.set_xlim(lo, hi)
        ax_sc.set_ylim(lo, hi)
        ax_sc.set_aspect("equal")
        ax_sc.legend(fontsize=6.5, framealpha=0.9, loc="upper left",
                     bbox_to_anchor=(1.02, 1), borderaxespad=0)
    else:
        ax_sc.text(0.5, 0.5,
                   "No normalized data available.",
                   ha="center", va="center", transform=ax_sc.transAxes, fontsize=12)
        ax_sc.set_title("Normalized scatter — no data")

    # === Bottom: per-study r bars ==========================================
    x_pos      = np.arange(len(all_seq_ids))
    bar_h      = []
    bar_col    = []
    bar_hatch  = []
    bar_xlabs  = []

    for seq_id in all_seq_ids:
        r_val = study_r.get(seq_id)
        n_val = study_n.get(seq_id, 0)
        col   = palette.get(seq_id, "#BDBDBD")
        short = textwrap.shorten(study_label_map.get(seq_id, f"seq={seq_id}"), 20)

        if r_val is not None:
            bar_h.append(r_val)
            bar_col.append(col)
            bar_hatch.append("")
        else:
            bar_h.append(0.0)
            bar_col.append("#CCCCCC")
            bar_hatch.append("///")
        bar_xlabs.append(f"seq={seq_id}\n(n={n_val})")

    bar_objs = ax_bars.bar(x_pos, bar_h, color=bar_col, width=0.65,
                           edgecolor="white", zorder=2)
    for bo, hatch, col in zip(bar_objs, bar_hatch, bar_col):
        if hatch:
            bo.set_hatch(hatch)
            bo.set_edgecolor("#888888")

    ax_bars.axhline(0,    color="black", lw=0.9, zorder=3)
    ax_bars.axhline( 0.5, color="gray",  lw=0.5, ls="--", alpha=0.45)
    ax_bars.axhline(-0.5, color="gray",  lw=0.5, ls="--", alpha=0.45)
    ax_bars.set_xticks(x_pos)
    ax_bars.set_xticklabels(bar_xlabs, fontsize=7.5)
    ax_bars.set_ylabel("Within-study Pearson r", fontsize=10)
    ax_bars.set_ylim(-1.1, 1.1)
    ax_bars.spines[["top", "right"]].set_visible(False)
    ax_bars.set_title(
        "Per-study within-study Pearson r  "
        "(hatched = n < 2 comparable pairs or zero variance)",
        fontsize=9,
    )

    # Annotate Fisher z-weighted r
    if per_study_r_n:
        z_arr  = np.array([np.arctanh(np.clip(r, -0.9999, 0.9999))
                           for r, _ in per_study_r_n])
        w_arr  = np.array([n for _, n in per_study_r_n], dtype=float)
        r_fish = float(np.tanh(np.average(z_arr, weights=w_arr)))
        ax_bars.text(
            0.99, 0.97,
            f"Fisher z weighted r = {r_fish:.3f}\n"
            f"({len(per_study_r_n)} studies with r)",
            transform=ax_bars.transAxes, fontsize=8.5,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow",
                      ec="gray", alpha=0.85),
        )

    fig.suptitle(
        f"LLM Simulation vs Ground Truth — Aggregate Studies  ({args.config})"
        + (f"\n{title_suffix}" if title_suffix else ""),
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Collapsed-arm filter
# ---------------------------------------------------------------------------

def filter_collapsed(rows: list[dict]) -> list[dict]:
    """Return rows with comparable arms where the LLM answered at 0% or 100% removed.

    An arm×outcome is considered "collapsed" when llm_mean == 0.0 or == 1.0,
    meaning every simulated respondent gave the same binary answer and the LLM
    produced no variance whatsoever.  Non-comparable rows are kept untouched so
    the set of studies shown in the bars panel stays the same.
    """
    kept = []
    n_removed = 0
    for r in rows:
        if r["comparable"] and r["llm_mean"] is not None and r["llm_mean"] in (0.0, 1.0):
            n_removed += 1
            # Keep as non-comparable so the study still appears in the bars panel
            kept.append({**r, "comparable": False,
                         "note": (r.get("note") or "") + " [collapsed: llm_mean=0/1]"})
        else:
            kept.append(r)
    print(f"  Collapsed filter: removed {n_removed} arms where llm_mean ∈ {{0, 1}}")
    return kept


def filter_sound(rows: list[dict]) -> list[dict]:
    """Keep only rows belonging to studies with verified sound extraction.

    Excluded studies and their reasons:
      151 — visual matrix-puzzle outcomes (LLM cannot perform the task)
      164 — outcomes require clicking/listening to audio recordings
      169 — real-money dictator game; binary choices collapse to noise
      174 — question_text = None for all 15 outcomes; LLM answered blind
      176 — all arm texts are identical; no treatment variation in the prompt
    """
    kept = [r for r in rows if r["seq_id"] not in UNSOUND_STUDIES]
    excluded_ids = sorted({r["seq_id"] for r in rows if r["seq_id"] in UNSOUND_STUDIES})
    print(f"  Sound filter: excluded studies {excluded_ids}  "
          f"({len(rows) - len(kept)} rows removed)")
    return kept


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
    plot_combined(rows, palette)

    print("\nGenerating collapsed-excluded variant...")
    rows_filt = filter_collapsed(rows)
    plot_combined(
        rows_filt, palette,
        out_path=COMBINED_FILT_OUT,
        title_suffix="Arms where LLM responded at 0% or 100% rate excluded",
    )

    print("\nGenerating sound-studies-only variant...")
    rows_sound = filter_sound(rows)
    plot_combined(
        rows_sound, palette,
        out_path=COMBINED_SOUND_OUT,
        title_suffix=(
            "Sound studies only  "
            "(excl. seq 151/164/169/174/176: visual tasks, audio outcomes, "
            "missing question text, or identical arm texts)"
        ),
    )
