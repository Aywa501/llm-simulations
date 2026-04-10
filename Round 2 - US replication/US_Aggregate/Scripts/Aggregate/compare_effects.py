"""
compare_effects.py

Compares LLM-predicted treatment effects against ground-truth effects
extracted by extract_effects_gt.py.

  GT source  : Data/Ground_Truth/study_effects_gt.jsonl
               (one record per study, contains pre-computed Δ values)

  LLM source : Data/Simulation/aggregate_simulation_raw_{cfg}.jsonl
               (raw per-respondent outputs — arm means computed here)

For each study × treatment arm × outcome:

    predicted_effect = LLM_mean(treatment_arm) − LLM_mean(control_arm)
    observed_effect  = GT delta from paper extraction

Primary evaluation: Pearson r(predicted_effect, observed_effect)
Also reports: Spearman r, RMSE, sign accuracy, Fisher z weighted r.

Outputs:
    Data/Results/aggregate_effects_table_{cfg}.csv
    Data/Results/aggregate_effects_summary_{cfg}.txt

Usage:
    python compare_effects.py [--config no_reasoning] [--sound-only]
"""

import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths & args
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR.parents[1] / "Data"

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="no_reasoning")
parser.add_argument("--gt-file", default="study_data.jsonl",
                    metavar="FILE",
                    help="GT JSONL inside Data/Ground_Truth/ "
                         "(default: study_data.jsonl from extract_study_data.py; "
                         "legacy: study_effects_gt.jsonl)")
parser.add_argument("--sound-only", action="store_true",
                    help="Exclude studies with known data-quality issues "
                         "(seq 151, 164, 169, 174, 176)")
args = parser.parse_args()

GT_PATH  = DATA_DIR / "Ground_Truth" / args.gt_file
SIM_PATH = DATA_DIR / "Simulation"   / f"aggregate_simulation_raw_{args.config}.jsonl"
OUT_CSV  = DATA_DIR / "Results"      / f"aggregate_effects_table_{args.config}.csv"
OUT_TXT  = DATA_DIR / "Results"      / f"aggregate_effects_summary_{args.config}.txt"
(DATA_DIR / "Results").mkdir(exist_ok=True)

UNSOUND_STUDIES: set[int] = {151, 164, 169, 174, 176}

# ---------------------------------------------------------------------------
# Slugify — must match simulate_aggregate.py
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50]

# ---------------------------------------------------------------------------
# Load ground-truth effects
# ---------------------------------------------------------------------------

def load_gt_effects() -> dict[int, dict]:
    """Returns {seq_id: {(arm_id, outcome_id): delta_value}}"""
    gt: dict[int, dict] = {}
    labels: dict[int, str] = {}
    ctrl_arms: dict[int, str] = {}

    if not GT_PATH.exists():
        print(f"WARNING: GT effects file not found at {GT_PATH}")
        print("         Run Scripts/Setup/extract_effects_gt.py first.")
        return {}, {}, {}

    for line in open(GT_PATH):
        rec = json.loads(line)

        # Accept both:
        #   study_data.jsonl     → results_status, ground_truth.effects, instrument.control_arm_id
        #   study_effects_gt.jsonl → extract_status, effects, control_arm_id
        if "ground_truth" in rec:
            # New unified format from extract_study_data.py
            if rec.get("results_status") not in ("ok", "partial"):
                continue
            effects_list = rec.get("ground_truth", {}).get("effects", [])
            ctrl_arm_id  = rec.get("instrument", {}).get("control_arm_id", "")
        else:
            # Legacy format from extract_effects_gt.py
            if rec.get("extract_status") not in ("ok", "partial"):
                continue
            effects_list = rec.get("effects", [])
            ctrl_arm_id  = rec.get("control_arm_id", "")

        seq_id = rec["seq_id"]
        labels[seq_id]    = rec.get("title", f"seq={seq_id}")
        ctrl_arms[seq_id] = ctrl_arm_id
        gt[seq_id] = {}
        for e in effects_list:
            delta = e.get("delta")
            if delta is None:
                continue
            arm_id     = e.get("arm_id", "")
            outcome_id = e.get("outcome_id", "") or slugify(e.get("outcome_name", ""))
            gt[seq_id][(arm_id, outcome_id)] = {
                "delta":          delta,
                "treatment_mean": e.get("treatment_mean"),
                "control_mean":   e.get("control_mean"),
                "n_treatment":    e.get("n_treatment"),
                "n_control":      e.get("n_control"),
                "metric":         e.get("metric", ""),
                "outcome_name":   e.get("outcome_name", ""),
            }

    return gt, labels, ctrl_arms

# ---------------------------------------------------------------------------
# Load simulation arm means
# ---------------------------------------------------------------------------

def load_sim_means(path: Path) -> dict[tuple, float]:
    """Returns {(seq_id, arm_id, outcome_id): mean_response}"""
    sums:   dict[tuple, float] = defaultdict(float)
    counts: dict[tuple, int]   = defaultdict(int)
    for line in open(path):
        r = json.loads(line)
        if r["parse_ok"] and r["value"] is not None:
            key = (r["seq_id"], r["arm_id"], r["outcome_id"])
            sums[key]   += r["value"]
            counts[key] += 1
    return {k: sums[k] / counts[k] for k in sums}

# ---------------------------------------------------------------------------
# Build comparison rows
# ---------------------------------------------------------------------------

def build_effect_rows(gt: dict, labels: dict, ctrl_arms: dict,
                      sim_means: dict) -> list[dict]:
    rows = []

    for seq_id, gt_effects in sorted(gt.items()):
        if args.sound_only and seq_id in UNSOUND_STUDIES:
            continue

        ctrl_arm = ctrl_arms.get(seq_id, "")
        label    = labels.get(seq_id, f"seq={seq_id}")

        # LLM mean for the control arm across all outcomes
        ctrl_llm: dict[str, float] = {}
        for outcome_id in {k[1] for k in gt_effects}:
            v = sim_means.get((seq_id, ctrl_arm, outcome_id))
            if v is not None:
                ctrl_llm[outcome_id] = v

        for (arm_id, outcome_id), gt_entry in sorted(gt_effects.items()):
            gt_delta = gt_entry["delta"]
            ctrl_mean_llm = ctrl_llm.get(outcome_id)
            treat_mean_llm = sim_means.get((seq_id, arm_id, outcome_id))

            if ctrl_mean_llm is None or treat_mean_llm is None:
                llm_effect  = None
                comparable  = False
                note = ("llm control mean missing"
                        if ctrl_mean_llm is None
                        else "llm treatment mean missing")
            else:
                llm_effect = round(treat_mean_llm - ctrl_mean_llm, 4)
                comparable  = True
                note = ""

            rows.append({
                "seq_id":           seq_id,
                "study_label":      label,
                "arm_id":           arm_id,
                "outcome_id":       outcome_id,
                "outcome_name":     gt_entry["outcome_name"],
                "control_arm":      ctrl_arm,
                "gt_delta":         round(gt_delta, 4),
                "llm_effect":       llm_effect,
                "gt_treatment_mean":gt_entry["treatment_mean"],
                "gt_control_mean":  gt_entry["control_mean"],
                "gt_n_treatment":   gt_entry["n_treatment"],
                "gt_n_control":     gt_entry["n_control"],
                "metric":           gt_entry["metric"],
                "comparable":       comparable,
                "note":             note,
            })

    return rows

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def sign_accuracy(predicted: list[float], actual: list[float]) -> tuple[float, int, int]:
    pairs = [(p, a) for p, a in zip(predicted, actual) if a != 0 and p != 0]
    if not pairs:
        return float("nan"), 0, 0
    correct = sum(1 for p, a in pairs if (p > 0) == (a > 0))
    return correct / len(pairs), correct, len(pairs)


def rmse(predicted: list[float], actual: list[float]) -> float:
    diffs = [(p - a) ** 2 for p, a in zip(predicted, actual)]
    return float(np.sqrt(np.mean(diffs)))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gt, labels, ctrl_arms = load_gt_effects()
    if not gt:
        return

    sim_means = load_sim_means(SIM_PATH)
    rows      = build_effect_rows(gt, labels, ctrl_arms, sim_means)

    # Write CSV
    fieldnames = [
        "seq_id", "study_label", "arm_id", "outcome_id", "outcome_name",
        "control_arm", "gt_delta", "llm_effect",
        "gt_treatment_mean", "gt_control_mean",
        "gt_n_treatment", "gt_n_control", "metric",
        "comparable", "note",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {OUT_CSV}")

    # Comparable subset
    comp = [r for r in rows
            if r["comparable"]
            and r["gt_delta"]   is not None
            and r["llm_effect"] is not None]
    print(f"Comparable contrasts: {len(comp)} / {len(rows)}")

    lines = [
        f"Config      : {args.config}",
        f"Sound-only  : {args.sound_only}",
        f"GT file     : {GT_PATH.name}",
        f"Contrasts (total)      : {len(rows)}",
        f"Contrasts (comparable) : {len(comp)}",
        f"Studies with data      : {len({r['seq_id'] for r in comp})}",
        "",
    ]

    if len(comp) < 3:
        lines.append("Too few comparable contrasts for correlation statistics.")
        lines.append("Run extract_effects_gt.py to improve GT coverage.")
    else:
        gt_fx  = [r["gt_delta"]   for r in comp]
        llm_fx = [r["llm_effect"] for r in comp]

        r_p, p_p = stats.pearsonr(gt_fx, llm_fx)
        r_s, p_s = stats.spearmanr(gt_fx, llm_fx)
        rmse_val  = rmse(llm_fx, gt_fx)
        sacc, n_corr, n_nz = sign_accuracy(llm_fx, gt_fx)

        lines += [
            f"Pearson  r     = {r_p:+.3f}  (p={p_p:.3f})",
            f"Spearman r     = {r_s:+.3f}  (p={p_s:.3f})",
            f"RMSE           = {rmse_val:.4f}",
            f"Sign accuracy  = {sacc:.1%}  ({n_corr}/{n_nz} non-zero contrasts)",
            "",
            "Per-study breakdown:",
        ]

        by_study: dict[int, list] = defaultdict(list)
        for r in comp:
            by_study[r["seq_id"]].append(r)

        per_r_n: list[tuple[float, int]] = []

        for seq_id, s_rows in sorted(by_study.items()):
            gt_s  = [r["gt_delta"]   for r in s_rows]
            llm_s = [r["llm_effect"] for r in s_rows]
            label = s_rows[0]["study_label"][:55]
            sa, nc, nnz = sign_accuracy(llm_s, gt_s)

            if len(gt_s) >= 2 and np.std(gt_s) > 0 and np.std(llm_s) > 0:
                rr, pp = stats.pearsonr(gt_s, llm_s)
                per_r_n.append((rr, len(gt_s)))
                lines.append(
                    f"  seq={seq_id:>3}  {label:<55}  "
                    f"n={len(gt_s):>2}  r={rr:+.3f}  p={pp:.3f}  "
                    f"sign={sa:.0%}"
                )
            else:
                lines.append(
                    f"  seq={seq_id:>3}  {label:<55}  "
                    f"n={len(gt_s):>2}  (too few / no variance)  "
                    f"sign={sa:.0%}"
                )

        if per_r_n:
            z_arr  = np.array([np.arctanh(np.clip(r, -0.9999, 0.9999))
                               for r, _ in per_r_n])
            w_arr  = np.array([n for _, n in per_r_n], dtype=float)
            r_fish = float(np.tanh(np.average(z_arr, weights=w_arr)))
            lines += [
                "",
                f"Fisher z-transform weighted r = {r_fish:+.3f}  "
                f"({len(per_r_n)} studies contributing)",
            ]

        # Excluded rows
        excluded = [r for r in rows if not r["comparable"]]
        if excluded:
            lines += ["", f"Not comparable ({len(excluded)} rows):"]
            for r in excluded:
                lines.append(
                    f"  seq={r['seq_id']:>3}  {r['arm_id']:<35}  "
                    f"{r['outcome_id']:<30}  {r['note']}"
                )

    summary = "\n".join(lines)
    print("\n" + summary)
    open(OUT_TXT, "w").write(summary + "\n")
    print(f"\nSummary → {OUT_TXT}")


if __name__ == "__main__":
    main()
