"""
compare_to_ground_truth.py

Loads simulation_raw.jsonl and SORTED DATA study_enriched JSONL,
applies per-study arm/outcome mappings, and produces:
  - comparison_table.csv  (one row per arm × outcome, LLM vs human mean)
  - comparison_summary.txt (correlation stats + notes)

Arm mapping notes
-----------------
seq=6  : sim arm_id = "{arm}__{outcome_id}" — split on "__" to recover both
seq=20 : sim outcome "belief_pct", E1 arms  → GT out1 (xenophobia belief)
         sim outcome "belief_pct", E2 arms  → GT out3 (Pittsburgh belief)
seq=28 : sim "belief_pct" → GT out1 ; sim "donation" → GT out2
seq=29 : sim "donation", arm1-arm4         → GT out1 (preferred)
seq=46 : arm label mismatch — marked NOT_COMPARABLE
seq=52 : arm labels match, GT out1         — ceiling effect flagged
seq=53 : arm labels don't match GT         — marked NOT_COMPARABLE
"""

import csv, json
from collections import defaultdict
from pathlib import Path
from scipy import stats

DATA_DIR = Path(__file__).resolve().parents[2] / "Data" / "Microdata"
SIM_PATH = DATA_DIR / "simulation_raw.jsonl"
GT_PATH  = DATA_DIR / "SORTED DATA - study_enriched_tier1and2_tagged.jsonl"
OUT_CSV  = DATA_DIR / "comparison_table.csv"
OUT_TXT  = DATA_DIR / "comparison_summary.txt"

# ---------------------------------------------------------------------------
# Study metadata for labelling
# ---------------------------------------------------------------------------
STUDY_LABELS = {
    6:  "Kidney allocation (morality/efficiency)",
    20: "Xenophobia & social desirability",
    28: "Islamophobia & social desirability",
    29: "Legitimacy & social desirability",
    46: "Hiring discrimination attitudes",
    52: "Job attraction (community demographics)",
    53: "Job attraction (performance bonus)",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_sim(path):
    """Returns {(seq_id, arm_id, outcome_id): [values]}"""
    sums = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r["parse_ok"] and r["value"] is not None:
            sums[(r["seq_id"], r["arm_id"], r["outcome_id"])].append(r["value"])
    return sums


def load_gt(path):
    """Returns {seq_id: {(arm_id, outcome_id): value}}"""
    gt = {}
    for line in open(path):
        r = json.loads(line)
        seq_id = r["seq_id"]
        outcomes = r["enrichment"]["results"]["outcomes"]
        gt[seq_id] = {}
        for o in outcomes:
            oid = o["outcome_id"]
            for gs in o["group_summaries"]:
                if gs["value"] is not None:
                    gt[seq_id][(gs["arm_id"], oid)] = {
                        "value": gs["value"],
                        "n":     gs["n_analyzed"],
                    }
    return gt


# ---------------------------------------------------------------------------
# Mapping rules
# ---------------------------------------------------------------------------

def get_comparable_pairs(sim_sums, gt):
    """
    Returns list of dicts, one per comparable (seq, arm, outcome) pair.
    """
    rows = []

    def add(seq_id, sim_arm, sim_out, gt_arm, gt_out, note="", comparable=True):
        sim_key = (seq_id, sim_arm, sim_out)
        gt_key  = (gt_arm, gt_out)
        sim_vals = sim_sums.get(sim_key, [])
        gt_entry = gt.get(seq_id, {}).get(gt_key)
        if not sim_vals:
            return
        llm_mean = sum(sim_vals) / len(sim_vals)
        human_mean = gt_entry["value"] if gt_entry else None
        human_n    = gt_entry["n"]     if gt_entry else None
        rows.append({
            "seq_id":      seq_id,
            "study_label": STUDY_LABELS[seq_id],
            "sim_arm_id":  sim_arm,
            "sim_out_id":  sim_out,
            "gt_arm_id":   gt_arm,
            "gt_out_id":   gt_out,
            "llm_mean":    round(llm_mean, 4),
            "llm_n":       len(sim_vals),
            "human_mean":  round(human_mean, 4) if human_mean is not None else None,
            "human_n":     human_n,
            "comparable":  comparable,
            "note":        note,
        })

    # ── seq=6 ──────────────────────────────────────────────────────────────
    # sim arm_id = "{arm}__{outcome_id}" ; outcome_id is always "support"
    for (seq, arm_id, out_id), vals in sim_sums.items():
        if seq != 6:
            continue
        parts  = arm_id.split("__")
        gt_arm = "__".join(parts[:-1])
        gt_out = parts[-1]
        add(6, arm_id, out_id, gt_arm, gt_out)

    # ── seq=20 ─────────────────────────────────────────────────────────────
    e1_arms = {"E1_Ctrl_Private", "E1_Ctrl_Public",
               "E1_TreatProb_Private", "E1_TreatProb_Public"}
    e2_arms = {"E2_ClintonWin_Private", "E2_ClintonWin_Public",
               "E2_TrumpWin_Private",   "E2_TrumpWin_Public"}
    for arm in e1_arms:
        add(20, arm, "belief_pct", arm, "out1")
    for arm in e2_arms:
        add(20, arm, "belief_pct", arm, "out3")

    # ── seq=28 ─────────────────────────────────────────────────────────────
    for arm in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        add(28, arm, "belief_pct", arm, "out1")
        add(28, arm, "donation",   arm, "out2",
            note="LLM safety refusal: always 0", comparable=False)

    # ── seq=29 ─────────────────────────────────────────────────────────────
    for arm in ["arm1", "arm2", "arm3", "arm4"]:
        add(29, arm, "donation", arm, "out1")

    # ── seq=46 ─────────────────────────────────────────────────────────────
    for arm in ["Control", "Same_Effort", "Same_Suffering",
                "No_Discrimination", "All_Constant"]:
        add(46, arm, "attitude", arm, "out1",
            note="Arm label mismatch: sim arms differ from microdata arms",
            comparable=False)

    # ── seq=52 ─────────────────────────────────────────────────────────────
    for arm in ["mostly_african_american", "mostly_hispanic",
                "mostly_white", "multiracial"]:
        add(52, arm, "job_attraction", arm, "out1",
            note="LLM ceiling effect: always 1.0")

    # ── seq=53 ─────────────────────────────────────────────────────────────
    for arm in ["bonus_large", "bonus_mod", "bonus_small", "no_bonus"]:
        add(53, arm, "job_attraction", arm, "out1",
            note="Arm label mismatch: sim arms differ from microdata arms",
            comparable=False)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sim_sums = load_sim(SIM_PATH)
    gt       = load_gt(GT_PATH)
    rows     = get_comparable_pairs(sim_sums, gt)

    # write CSV
    fieldnames = ["seq_id", "study_label", "sim_arm_id", "sim_out_id",
                  "gt_arm_id", "gt_out_id",
                  "llm_mean", "llm_n", "human_mean", "human_n",
                  "comparable", "note"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {OUT_CSV}")

    # correlation on fully comparable rows with human data
    comp = [r for r in rows
            if r["comparable"] and r["human_mean"] is not None and r["llm_mean"] is not None]
    print(f"\nComparable pairs (for correlation): {len(comp)}")

    lines = []
    if len(comp) >= 3:
        human_vals = [r["human_mean"] for r in comp]
        llm_vals   = [r["llm_mean"]   for r in comp]
        r_pearson, p_pearson   = stats.pearsonr(human_vals, llm_vals)
        r_spearman, p_spearman = stats.spearmanr(human_vals, llm_vals)
        lines += [
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
            if len(h) >= 2:
                rr, pp = stats.pearsonr(h, l)
                lines.append(f"  seq={seq_id} {STUDY_LABELS[seq_id][:50]:<50}  n={len(h)}  r={rr:.3f}  p={pp:.3f}")
            else:
                lines.append(f"  seq={seq_id} {STUDY_LABELS[seq_id][:50]:<50}  n={len(h)}  (too few for r)")
        lines += [
            "",
            "Excluded / flagged:",
        ]
        for r in rows:
            if not r["comparable"] or r["human_mean"] is None:
                lines.append(f"  seq={r['seq_id']} {r['sim_arm_id']:<40} {r['note']}")
    else:
        lines.append("Too few comparable pairs for correlation.")

    summary = "\n".join(lines)
    print("\n" + summary)
    open(OUT_TXT, "w").write(summary + "\n")
    print(f"\nSummary → {OUT_TXT}")


if __name__ == "__main__":
    main()
