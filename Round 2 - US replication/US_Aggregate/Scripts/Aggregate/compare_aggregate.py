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
from difflib import SequenceMatcher
from pathlib import Path
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parents[1] / "Data"

# Default GT file is the clean aligned extraction; fall back to old file if needed.
_DEFAULT_GT = "study_gt_v2.jsonl"

MAPPING_PATH = DATA_DIR / "id_mapping.json"

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="no_reasoning",
                    help="Batch config name (e.g. no_reasoning, reasoning_low, reasoning_medium)")
parser.add_argument("--gt-file", default=_DEFAULT_GT, metavar="FILE",
                    help=f"GT JSONL filename inside Data/Ground_Truth/ "
                         f"(default: {_DEFAULT_GT})")
parser.add_argument("--mapping", action="store_true",
                    help="Apply id_mapping.json produced by match_ids_llm.py "
                         "(only needed with the old GT file)")
args = parser.parse_args()

ENRICHED = DATA_DIR / "Ground_Truth" / args.gt_file

SIM_PATH = DATA_DIR / f"aggregate_simulation_raw_{args.config}.jsonl"
OUT_CSV  = DATA_DIR / "Results" / f"aggregate_comparison_table_{args.config}.csv"
OUT_TXT  = DATA_DIR / "Results" / f"aggregate_comparison_summary_{args.config}.txt"
(DATA_DIR / "Results").mkdir(exist_ok=True)

# Load LLM-produced ID mapping if requested
id_mapping: dict = {}
if args.mapping:
    if MAPPING_PATH.exists():
        id_mapping = json.loads(MAPPING_PATH.read_text())
        print(f"Loaded ID mapping from {MAPPING_PATH.name}  "
              f"({len(id_mapping)} studies covered)")
    else:
        print(f"WARNING: --mapping requested but {MAPPING_PATH.name} not found. "
              f"Run match_ids_llm.py first.")

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

def _slug_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _best_outcome_match(sim_outcome_id: str,
                        gt_outcome_ids: list[str],
                        threshold: float = 0.60) -> tuple[str | None, float]:
    """Return the best-matching GT outcome slug and its similarity score,
    or (None, 0) if nothing clears the threshold."""
    best_slug, best_score = None, 0.0
    for gt_slug in gt_outcome_ids:
        score = _slug_similarity(sim_outcome_id, gt_slug)
        if score > best_score:
            best_slug, best_score = gt_slug, score
    if best_score >= threshold:
        return best_slug, best_score
    return None, 0.0


def load_gt(path: Path) -> dict[int, dict]:
    """
    Returns (
        {seq_id: {(arm_id, outcome_id): {value, n, metric, sd, se}}},
        {seq_id: study_label},
        {seq_id: [gt_outcome_ids]},   # for fuzzy matching in build_rows
    )
    """
    gt: dict[int, dict] = {}
    labels: dict[int, str] = {}
    gt_outcome_ids: dict[int, list[str]] = {}

    for line in open(path):
        rec = json.loads(line)
        if rec.get("extract_status") not in ("ok", "partial"):
            continue
        seq_id = rec["seq_id"]
        labels[seq_id] = rec.get("title", f"seq={seq_id}")

        results    = rec.get("results") or {}
        instrument = rec.get("instrument") or {}

        # Map instrument outcome_question slugs → results outcome slugs
        # (positional, when counts match) so simulation outcome IDs derived
        # from the instrument can still find GT values from the results section.
        out_questions  = instrument.get("outcome_questions", [])
        result_outcomes = results.get("outcomes", [])
        instr_slugs    = [_slugify(q.get("outcome_name", "")) for q in out_questions]
        result_slugs   = [_slugify(o.get("name", ""))         for o in result_outcomes]

        # Build positional alias: instrument_slug → result_slug
        alias: dict[str, str] = {}
        if len(instr_slugs) == len(result_slugs):
            for i_slug, r_slug in zip(instr_slugs, result_slugs):
                if i_slug and r_slug and i_slug != r_slug:
                    alias[i_slug] = r_slug

        gt[seq_id] = {}
        for outcome in result_outcomes:
            raw_name   = outcome.get("name", "")
            outcome_id = _slugify(raw_name)

            for gs in outcome.get("group_summaries", []):
                arm_id = gs.get("arm_id")
                value  = gs.get("value")
                if arm_id and value is not None:
                    entry = {
                        "value":  value,
                        "n":      gs.get("n_analyzed"),
                        "metric": gs.get("metric"),
                        "sd":     gs.get("sd"),
                        "se":     gs.get("se"),
                    }
                    gt[seq_id][(arm_id, outcome_id)] = entry
                    # Also index under any instrument alias for this outcome
                    for i_slug, r_slug in alias.items():
                        if r_slug == outcome_id:
                            gt[seq_id][(arm_id, i_slug)] = entry

        gt_outcome_ids[seq_id] = list({k[1] for k in gt[seq_id]})

    return gt, labels, gt_outcome_ids


def _slugify(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50]

# ---------------------------------------------------------------------------
# Build comparison rows
# ---------------------------------------------------------------------------

def apply_id_mapping(seq_id: int, arm_id: str, outcome_id: str) -> tuple[str, str, str]:
    """
    Returns (mapped_arm_id, mapped_outcome_id, mapping_note).
    Consults id_mapping (populated from id_mapping.json when --mapping is used).
    A null value in the mapping means the LLM determined there is no GT match.
    """
    study_map = id_mapping.get(str(seq_id), {})
    arm_map   = study_map.get("arms",     {})
    out_map   = study_map.get("outcomes", {})

    note = ""
    mapped_arm = arm_id
    mapped_out = outcome_id

    if arm_id in arm_map:
        mapped = arm_map[arm_id]
        if mapped is None:
            return None, None, "arm explicitly unmapped (no GT equivalent)"
        mapped_arm = mapped
        if mapped_arm != arm_id:
            note = f"arm_mapped:{arm_id}→{mapped_arm}"

    if outcome_id in out_map:
        mapped = out_map[outcome_id]
        if mapped is None:
            return None, None, "outcome explicitly unmapped (no GT equivalent)"
        mapped_out = mapped
        if mapped_out != outcome_id:
            sep  = "; " if note else ""
            note = note + sep + f"outcome_mapped:{outcome_id}→{mapped_out}"

    return mapped_arm, mapped_out, note


def build_rows(sim_sums, gt, labels, gt_outcome_ids) -> list[dict]:
    rows = []

    for (seq_id, arm_id, outcome_id), sim_vals in sorted(sim_sums.items()):
        if not sim_vals:
            continue

        llm_mean = sum(sim_vals) / len(sim_vals)
        study_gt = gt.get(seq_id, {})

        # 1) Apply LLM id_mapping (if --mapping flag was used)
        mapped_arm, mapped_out, map_note = apply_id_mapping(seq_id, arm_id, outcome_id)
        if mapped_arm is None:
            # Explicitly unmapped by the LLM — skip lookup
            rows.append({
                "seq_id":      seq_id,
                "study_label": labels.get(seq_id, f"seq={seq_id}"),
                "arm_id":      arm_id,
                "outcome_id":  outcome_id,
                "llm_mean":    round(llm_mean, 4),
                "llm_n":       len(sim_vals),
                "human_mean":  None,
                "human_n":     None,
                "comparable":  False,
                "note":        map_note,
            })
            continue

        # 2) Exact match (using mapped IDs)
        gt_entry   = study_gt.get((mapped_arm, mapped_out))
        note       = map_note
        comparable = True

        # 3) Fuzzy outcome match as final fallback (if no mapping and exact fails)
        if gt_entry is None and not map_note:
            fuzzy_slug, fuzzy_score = _best_outcome_match(
                mapped_out, gt_outcome_ids.get(seq_id, [])
            )
            if fuzzy_slug is not None:
                gt_entry = study_gt.get((mapped_arm, fuzzy_slug))
                if gt_entry is not None:
                    note = (f"fuzzy_outcome_match:{fuzzy_slug}"
                            f"(sim={mapped_out},score={fuzzy_score:.2f})")

        if gt_entry is None:
            has_any_gt = any(k[1] == mapped_out for k in study_gt)
            note = ("outcome not found in GT extraction"
                    if not has_any_gt else "arm not found in GT extraction")
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
    sim_sums                  = load_sim(SIM_PATH)
    gt, labels, gt_outcome_ids = load_gt(ENRICHED)
    rows                      = build_rows(sim_sums, gt, labels, gt_outcome_ids)

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
