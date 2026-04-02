"""
compare_aggregate.py

Loads aggregate_simulation_raw_{cfg}.jsonl and study_enriched_aggregate.jsonl,
matches LLM arm means to extracted ground-truth arm means, and produces:

  aggregate_comparison_table_{cfg}.csv   — one row per arm × outcome
  aggregate_comparison_summary_{cfg}.txt — correlation statistics

Arm matching is generic: arm_ids are shared between the simulation output
and the enriched JSONL (both derived from the same slugified arm labels),
so no per-study hardcoding is required.

Studies flagged as not comparable (missing GT value, scale mismatch, etc.)
are included in the CSV but excluded from the correlation.

Usage:
    python compare_aggregate.py [--config no_reasoning]
"""

import argparse, csv, json
from collections import defaultdict
from pathlib import Path
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parents[1] / "Data"
ENRICHED   = DATA_DIR / "study_enriched_aggregate.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="no_reasoning",
                    help="Batch config name (e.g. no_reasoning, reasoning_low, reasoning_medium)")
args = parser.parse_args()

SIM_PATH = DATA_DIR / f"aggregate_simulation_raw_{args.config}.jsonl"
OUT_CSV  = DATA_DIR / f"aggregate_comparison_table_{args.config}.csv"
OUT_TXT  = DATA_DIR / f"aggregate_comparison_summary_{args.config}.txt"

print(f"Config: {args.config}")
print(f"Sim    : {SIM_PATH.name}")
print(f"GT     : {ENRICHED.name}")

# ---------------------------------------------------------------------------
# Load simulation results
# ---------------------------------------------------------------------------

def load_sim(path: Path) -> dict[tuple, list]:
    """Returns {(seq_id, arm_id, outcome_id): [values]}"""
    sums: dict[tuple, list] = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r["parse_ok"] and r["value"] is not None:
            sums[(r["seq_id"], r["arm_id"], r["outcome_id"])].append(r["value"])
    return sums

# ---------------------------------------------------------------------------
# Load ground truth from enriched JSONL
# ---------------------------------------------------------------------------

def load_gt(path: Path) -> dict[int, dict]:
    """
    Returns {seq_id: {(arm_id, outcome_id): {value, n, metric, unit}}}
    """
    gt: dict[int, dict] = {}
    labels: dict[int, str] = {}

    for line in open(path):
        rec = json.loads(line)
        if rec.get("extract_status") != "ok":
            continue
        seq_id = rec["seq_id"]
        labels[seq_id] = rec.get("title", f"seq={seq_id}")

        results    = rec.get("results") or {}
        instrument = rec.get("instrument") or {}

        # Build outcome_id lookup: slugified outcome_name → outcome metadata
        out_questions = instrument.get("outcome_questions", [])
        out_id_map: dict[str, str] = {}
        for q in out_questions:
            raw_name = q.get("outcome_name", "")
            slug_id  = _slugify(raw_name)
            out_id_map[raw_name] = slug_id

        gt[seq_id] = {}
        for outcome in results.get("outcomes", []):
            raw_name   = outcome.get("name", "")
            outcome_id = _slugify(raw_name)

            for gs in outcome.get("group_summaries", []):
                arm_id = gs.get("arm_id")
                value  = gs.get("value")
                if arm_id and value is not None:
                    gt[seq_id][(arm_id, outcome_id)] = {
                        "value":  value,
                        "n":      gs.get("n_analyzed"),
                        "metric": gs.get("metric"),
                        "sd":     gs.get("sd"),
                        "se":     gs.get("se"),
                    }

    return gt, labels


def _slugify(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50]

# ---------------------------------------------------------------------------
# Build comparison rows
# ---------------------------------------------------------------------------

def build_rows(sim_sums, gt, labels) -> list[dict]:
    rows = []

    # Collect all (seq_id, arm_id, outcome_id) from simulation
    for (seq_id, arm_id, outcome_id), sim_vals in sorted(sim_sums.items()):
        if not sim_vals:
            continue

        llm_mean   = sum(sim_vals) / len(sim_vals)
        gt_entry   = (gt.get(seq_id) or {}).get((arm_id, outcome_id))
        study_gt   = gt.get(seq_id, {})

        # Determine comparability
        note       = ""
        comparable = True

        if gt_entry is None:
            # Try to find if any GT arm matches at all for this outcome
            has_any_gt = any(k[1] == outcome_id for k in study_gt)
            if not has_any_gt:
                note = "outcome not found in GT extraction"
            else:
                note = "arm not found in GT extraction"
            comparable = False

        human_mean = gt_entry["value"] if gt_entry else None
        human_n    = gt_entry["n"]     if gt_entry else None

        rows.append({
            "seq_id":      seq_id,
            "study_label": labels.get(seq_id, f"seq={seq_id}"),
            "arm_id":      arm_id,
            "outcome_id":  outcome_id,
            "llm_mean":    round(llm_mean, 4),
            "llm_n":       len(sim_vals),
            "human_mean":  round(human_mean, 4) if human_mean is not None else None,
            "human_n":     human_n,
            "comparable":  comparable,
            "note":        note,
        })

    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sim_sums       = load_sim(SIM_PATH)
    gt, labels     = load_gt(ENRICHED)
    rows           = build_rows(sim_sums, gt, labels)

    # Write CSV
    fieldnames = ["seq_id", "study_label", "arm_id", "outcome_id",
                  "llm_mean", "llm_n", "human_mean", "human_n",
                  "comparable", "note"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {OUT_CSV}")

    # Correlation on fully comparable rows with both LLM and human values
    comp = [r for r in rows
            if r["comparable"]
            and r["human_mean"] is not None
            and r["llm_mean"]   is not None]
    print(f"Comparable pairs (for correlation): {len(comp)}")

    lines = []
    if len(comp) >= 3:
        human_vals = [r["human_mean"] for r in comp]
        llm_vals   = [r["llm_mean"]   for r in comp]
        r_pearson,  p_pearson  = stats.pearsonr(human_vals, llm_vals)
        r_spearman, p_spearman = stats.spearmanr(human_vals, llm_vals)
        lines += [
            f"Config: {args.config}",
            f"Comparable arm×outcome pairs : {len(comp)}",
            f"Pearson  r = {r_pearson:.3f}  (p={p_pearson:.3f})",
            f"Spearman r = {r_spearman:.3f}  (p={p_spearman:.3f})",
            "",
            "Per-study breakdown:",
        ]
        by_study = defaultdict(list)
        for r in comp:
            by_study[r["seq_id"]].append(r)

        for seq_id, study_rows in sorted(by_study.items()):
            h = [r["human_mean"] for r in study_rows]
            l = [r["llm_mean"]   for r in study_rows]
            label = labels.get(seq_id, f"seq={seq_id}")[:55]
            if len(h) >= 2:
                rr, pp = stats.pearsonr(h, l)
                lines.append(
                    f"  seq={seq_id:>3}  {label:<55}  n={len(h)}  r={rr:.3f}  p={pp:.3f}"
                )
            else:
                lines.append(
                    f"  seq={seq_id:>3}  {label:<55}  n={len(h)}  (too few for r)"
                )

        lines += ["", "Excluded / flagged:"]
        for r in rows:
            if not r["comparable"] or r["human_mean"] is None:
                lines.append(
                    f"  seq={r['seq_id']:>3}  {r['arm_id']:<40}  {r['outcome_id']:<20}  {r['note']}"
                )
    else:
        lines.append(f"Too few comparable pairs for correlation (n={len(comp)}).")
        lines.append("Run extract_from_paper.py to improve GT extraction coverage.")

    summary = "\n".join(lines)
    print("\n" + summary)
    open(OUT_TXT, "w").write(summary + "\n")
    print(f"\nSummary → {OUT_TXT}")


if __name__ == "__main__":
    main()
