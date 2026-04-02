"""
inject_manual_extractions.py

Injects manually extracted paper data (from LLM chat interface) into
study_enriched_aggregate.jsonl, overwriting any previous extraction for
the same seq_id.

Input:  a JSON file or stdin containing an array of extraction objects
        with the structure returned by the manual extraction prompt.

Usage:
    python inject_manual_extractions.py extractions.json
    cat extractions.json | python inject_manual_extractions.py
    python inject_manual_extractions.py extractions.json --dry-run
"""

import argparse, json, re, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parents[1] / "Data"
ENRICHED   = DATA_DIR / "study_enriched_aggregate.jsonl"

# ---------------------------------------------------------------------------
# Helpers (mirrors extract_from_paper.py)
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50]


def normalise_extraction(raw: dict) -> dict:
    """
    Convert the manual extraction format into the instrument/results schema
    used by study_enriched_aggregate.jsonl.
    """
    # Ensure arm_ids are set and consistent
    variations = raw.get("treatment_variations", [])
    seen: dict[str, int] = {}
    for v in variations:
        if not v.get("arm_id"):
            base = slugify(v.get("arm_label", "arm"))
            if base in seen:
                seen[base] += 1
                v["arm_id"] = f"{base}_{seen[base]}"
            else:
                seen[base] = 0
                v["arm_id"] = base

    # Build arm_label → arm_id map for result matching
    label_map = {v["arm_label"].lower().strip(): v["arm_id"] for v in variations}

    # Assign arm_ids to results rows
    results_rows = raw.get("results", [])
    for r in results_rows:
        if not r.get("arm_id"):
            r["arm_id"] = label_map.get(
                r.get("arm_label", "").lower().strip(),
                slugify(r.get("arm_label", "arm"))
            )

    # Convert flat results list into enriched outcomes format
    outcomes_map: dict[str, dict] = {}
    for r in results_rows:
        out_name = r.get("outcome_name", "outcome")
        if out_name not in outcomes_map:
            outcomes_map[out_name] = {
                "name":           out_name,
                "is_primary":     True,
                "outcome_type":   "continuous",
                "table_reference": None,
                "group_summaries": [],
            }
        outcomes_map[out_name]["group_summaries"].append({
            "arm_label":   r.get("arm_label"),
            "arm_id":      r.get("arm_id"),
            "n_analyzed":  r.get("n"),
            "metric":      "mean",
            "value":       r.get("value"),
            "sd":          r.get("sd"),
            "se":          r.get("se"),
            "ci_lower":    None,
            "ci_upper":    None,
        })

    instrument = {
        "found":                bool(variations),
        "preamble":             raw.get("preamble"),
        "treatment_variations": variations,
        "outcome_questions":    raw.get("outcome_questions", []),
        "source_location":      "manual extraction",
    }

    results = {
        "outcomes":          list(outcomes_map.values()),
        "extraction_notes":  "manually extracted via chat interface",
    }

    return {"instrument": instrument, "results": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="-",
                        help="JSON file with extraction array (default: stdin)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing")
    args = parser.parse_args()

    # Read input
    if args.input == "-":
        raw_text = sys.stdin.read()
    else:
        raw_text = Path(args.input).read_text()

    extractions = json.loads(raw_text)
    if isinstance(extractions, dict):
        extractions = [extractions]  # single object wrapped in list

    print(f"Injecting {len(extractions)} extraction(s)…")

    # Load existing enriched file
    existing: dict[int, dict] = {}
    if ENRICHED.exists():
        for line in ENRICHED.open():
            r = json.loads(line)
            existing[r["seq_id"]] = r

    # Apply each extraction
    for ext in extractions:
        seq_id = int(ext["seq_id"])
        norm   = normalise_extraction(ext)

        if seq_id not in existing:
            print(f"  WARNING seq={seq_id} not in enriched file — creating stub")
            existing[seq_id] = {"seq_id": seq_id, "extract_status": "manual"}

        existing[seq_id]["extract_status"] = "ok"
        existing[seq_id]["instrument"]     = norm["instrument"]
        existing[seq_id]["results"]        = norm["results"]

        n_arms = len(ext.get("treatment_variations", []))
        n_out  = len(ext.get("outcome_questions", []))
        n_res  = len(ext.get("results", []))
        print(f"  seq={seq_id:>3}  arms={n_arms}  outcomes={n_out}  result_rows={n_res}  "
              f"{existing[seq_id].get('title','')[:50]}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    with open(ENRICHED, "w") as f:
        for r in sorted(existing.values(), key=lambda x: x["seq_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote → {ENRICHED}")


if __name__ == "__main__":
    main()
