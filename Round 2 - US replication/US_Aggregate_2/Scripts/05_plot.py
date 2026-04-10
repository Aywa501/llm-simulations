"""
05_plot.py  —  Step 5 of the pipeline

Generates a two-panel figure from the effects table produced by
04_compare_effects.py:

  Top panel   — scatter of LLM-predicted treatment effects vs ground-truth
                treatment effects.  One point per study × arm × outcome.
                Points coloured by study; diagonal = perfect prediction.

  Bottom panel — per-study Pearson r bars (computed on treatment effects).
                 Annotated with Fisher z-transform weighted r.

Reads  : Data/Results/effects_table_{cfg}.csv
Writes : Figures/effects_{cfg}.pdf

Usage:
    python 05_plot.py [--config no_reasoning] [--sound-only]
"""

import argparse, csv, textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths & args
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "Data"
FIGS_DIR   = SCRIPT_DIR.parent / "Figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--sound-only", action="store_true",
                    help="Exclude studies with known data-quality issues")
parser.add_argument("--normalize", action="store_true",
                    help="Divide effects by (scale_max - scale_min) before "
                         "plotting; rows with unknown scale bounds are excluded")
args = parser.parse_args()

CSV_PATH    = DATA_DIR / "Results" / "effects_table.csv"
OUT_PDF_ALL = FIGS_DIR / "effects.pdf"
OUT_PDF_NOS = FIGS_DIR / "effects_drop_one_sided.pdf"

# Studies excluded under --sound-only:
#   151 — visual matrix-puzzle (LLM cannot see the image)
#   164 — outcomes require clicking/listening to audio recordings
#   169 — real-money dictator game; binary choices produce no signal
#   174 — question_text = None for all outcomes; LLM answered blind
#   176 — all arm texts are identical; no treatment variation in prompt
UNSOUND_STUDIES: set[int] = {151, 164, 169, 174, 176}

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

def load_effects() -> list[dict]:
    if not CSV_PATH.exists():
        print(f"Effects CSV not found: {CSV_PATH}")
        print("Run 04_compare_effects.py first.")
        return []

    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            r["seq_id"]     = int(r["seq_id"])
            r["comparable"] = r["comparable"] == "True"
            r["one_sided"]  = r.get("one_sided", "False") == "True"
            for k in ("gt_delta", "llm_effect"):
                r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
            for k in ("scale_min", "scale_max"):
                v = r.get(k)
                r[k] = float(v) if v not in (None, "", "None") else None
            rows.append(r)
    return rows


def apply_normalization(rows: list[dict]) -> list[dict]:
    """Divide gt_delta and llm_effect by (scale_max - scale_min).
    Rows where the range is unknown or zero are marked not comparable."""
    out = []
    excluded = 0
    for r in rows:
        r = dict(r)
        smin, smax = r.get("scale_min"), r.get("scale_max")
        rng = (smax - smin) if (smin is not None and smax is not None) else None
        if rng and rng != 0:
            if r["gt_delta"] is not None:
                r["gt_delta"] = round(r["gt_delta"] / rng, 6)
            if r["llm_effect"] is not None:
                r["llm_effect"] = round(r["llm_effect"] / rng, 6)
        else:
            r["comparable"] = False
            excluded += 1
        out.append(r)
    if excluded:
        print(f"Normalization: {excluded} rows excluded (unknown scale range)")
    return out

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_effects(rows: list[dict], palette: dict, out_pdf: Path, subtitle: str = ""):
    comp = [r for r in rows
            if r["comparable"]
            and r["gt_delta"]   is not None
            and r["llm_effect"] is not None]

    if not comp:
        print("No comparable contrasts — nothing to plot.")
        return

    all_seq_ids     = sorted({r["seq_id"] for r in comp})
    study_label_map = {r["seq_id"]: r["study_label"] for r in rows}
    by_study: dict[int, list] = defaultdict(list)
    for r in comp:
        by_study[r["seq_id"]].append(r)

    gt_fx  = np.array([r["gt_delta"]   for r in comp])
    llm_fx = np.array([r["llm_effect"] for r in comp])
    r_all, p_all = stats.pearsonr(gt_fx, llm_fx)

    n_nonzero = sum(1 for g, l in zip(gt_fx, llm_fx) if g != 0 and l != 0)
    n_correct  = sum(1 for g, l in zip(gt_fx, llm_fx)
                     if g != 0 and l != 0 and (g > 0) == (l > 0))
    sign_acc   = n_correct / n_nonzero if n_nonzero else float("nan")

    # Per-study r
    study_r: dict[int, float | None] = {}
    study_n: dict[int, int]          = {}
    per_r_n: list[tuple[float, int]] = []

    for seq_id in all_seq_ids:
        s     = by_study[seq_id]
        study_n[seq_id] = len(s)
        gt_s  = np.array([r["gt_delta"]   for r in s])
        llm_s = np.array([r["llm_effect"] for r in s])
        if len(s) >= 2 and gt_s.std() > 0 and llm_s.std() > 0:
            rr, _ = stats.pearsonr(gt_s, llm_s)
            study_r[seq_id] = float(rr)
            per_r_n.append((float(rr), len(s)))
        else:
            study_r[seq_id] = None

    # ---- Layout ------------------------------------------------------------
    fig = plt.figure(figsize=(13, 11))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.85], hspace=0.45)
    ax_sc   = fig.add_subplot(gs[0])
    ax_bars = fig.add_subplot(gs[1])

    # === Top: effects scatter ===============================================
    pad = max(abs(gt_fx).max(), abs(llm_fx).max()) * 0.15
    lo  = min(gt_fx.min(), llm_fx.min()) - pad
    hi  = max(gt_fx.max(), llm_fx.max()) + pad

    ax_sc.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax_sc.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax_sc.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.35,
               label="Perfect prediction")

    seen: set[int] = set()
    for r in comp:
        sid  = r["seq_id"]
        col  = palette.get(sid, "#888888")
        short = textwrap.shorten(r["study_label"], 32)
        lab   = f"seq={sid}: {short}" if sid not in seen else None
        seen.add(sid)
        ax_sc.scatter(r["gt_delta"], r["llm_effect"],
                      c=col, s=55, alpha=0.85, edgecolors="white", lw=0.3,
                      label=lab, zorder=3)

    m, b = np.polyfit(gt_fx, llm_fx, 1)
    x_fit = np.linspace(lo, hi, 200)
    ax_sc.plot(x_fit, m * x_fit + b, color="#333", lw=1.2, alpha=0.6,
               label=f"OLS fit  r={r_all:.2f}, p={p_all:.3f}")

    norm_sfx = "  [normalized to scale range]" if args.normalize else ""
    ax_sc.set_xlabel(f"GT treatment effect  (Δ = treatment − control){norm_sfx}", fontsize=11)
    ax_sc.set_ylabel(f"LLM predicted effect  (Δ = treatment − control){norm_sfx}", fontsize=11)
    ax_sc.set_title(
        f"LLM-predicted vs observed treatment effects — gpt-4.1\n"
        f"n={len(comp)} contrasts  |  sign accuracy = {sign_acc:.0%}",
        fontsize=10,
    )
    ax_sc.set_xlim(lo, hi)
    ax_sc.set_ylim(lo, hi)
    ax_sc.set_aspect("equal")
    ax_sc.legend(fontsize=6.5, framealpha=0.9, loc="upper left",
                 bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # === Bottom: per-study r bars ==========================================
    x_pos     = np.arange(len(all_seq_ids))
    bar_h     = []
    bar_col   = []
    bar_hatch = []
    bar_xlabs = []

    for seq_id in all_seq_ids:
        r_val = study_r.get(seq_id)
        n_val = study_n.get(seq_id, 0)
        col   = palette.get(seq_id, "#CCCCCC")
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
    for bo, hatch in zip(bar_objs, bar_hatch):
        if hatch:
            bo.set_hatch(hatch)
            bo.set_edgecolor("#888888")

    ax_bars.axhline(0,    color="black", lw=0.9, zorder=3)
    ax_bars.axhline( 0.5, color="gray",  lw=0.5, ls="--", alpha=0.45)
    ax_bars.axhline(-0.5, color="gray",  lw=0.5, ls="--", alpha=0.45)
    ax_bars.set_xticks(x_pos)
    ax_bars.set_xticklabels(bar_xlabs, fontsize=7.5)
    ax_bars.set_ylabel("Per-study Pearson r  (on treatment effects)", fontsize=10)
    ax_bars.set_ylim(-1.1, 1.1)
    ax_bars.spines[["top", "right"]].set_visible(False)
    ax_bars.set_title(
        "Per-study r on treatment effects  "
        "(hatched = n < 2 contrasts or zero variance)",
        fontsize=9,
    )

    if per_r_n:
        z_arr  = np.array([np.arctanh(np.clip(r, -0.9999, 0.9999))
                           for r, _ in per_r_n])
        w_arr  = np.array([n for _, n in per_r_n], dtype=float)
        r_fish = float(np.tanh(np.average(z_arr, weights=w_arr)))
        ax_bars.text(
            0.99, 0.97,
            f"Fisher z weighted r = {r_fish:.3f}\n"
            f"({len(per_r_n)} studies with r)",
            transform=ax_bars.transAxes, fontsize=8.5,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow",
                      ec="gray", alpha=0.85),
        )

    sound_note = "  (sound studies only)" if args.sound_only else ""
    norm_note  = "  [normalized]" if args.normalize else ""
    sub_note   = f"  [{subtitle}]" if subtitle else ""
    fig.suptitle(
        f"LLM vs Ground Truth — Treatment Effects  (gpt-4.1){sound_note}{norm_note}{sub_note}",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"  Pearson r={r_all:.3f}  p={p_all:.3f}  "
          f"sign accuracy={sign_acc:.0%}  n={len(comp)}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = load_effects()
    if not rows:
        raise SystemExit(1)

    if args.sound_only:
        before = len(rows)
        rows = [r for r in rows if r["seq_id"] not in UNSOUND_STUDIES]
        excl = sorted({r["seq_id"] for r in load_effects()
                       if r["seq_id"] in UNSOUND_STUDIES})
        print(f"Sound filter: excluded {excl}  ({before - len(rows)} rows removed)")

    if args.normalize:
        rows = apply_normalization(rows)

    seq_ids = sorted({r["seq_id"] for r in rows})
    palette = build_palette(seq_ids)
    print(f"Loaded {len(rows)} rows from {CSV_PATH.name}  ({len(seq_ids)} studies)")

    # Full plot
    plot_effects(rows, palette, OUT_PDF_ALL)

    # Drop-one-sided plot
    rows_nos = [r for r in rows if not r["one_sided"]]
    n_dropped = len(rows) - len(rows_nos)
    print(f"One-sided filter: {n_dropped} rows dropped")
    plot_effects(rows_nos, palette, OUT_PDF_NOS, subtitle="drop one-sided")
