#!/usr/bin/env python3
"""
build_design_specs.py

Turn AEA RCT Registry-style trial JSON (dict-of-dicts) into prompt-ready
"design specs" JSONL: one line per trial.

Focus: experimental design (arms, outcomes, randomization, sample sizes),
NOT results.

Usage:
  python build_design_specs.py \
    --in data/trials_sampled_50.json \
    --out data/design_specs.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import norm_text, split_bullets, parse_bool


# ---------- helpers ----------

def get_first(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", [], {}):
            return d[k]
    return None


def parse_sample_sizes(trial: Dict[str, Any]) -> Dict[str, str]:
    # Keep as strings (registry uses mixed formats like "N/A", "2500 individuals/1600 individuals")
    return {
        "planned_clusters": norm_text(trial.get("Sample size: planned number of clusters")),
        "planned_observations": norm_text(trial.get("Sample size: planned number of observations")),
        "planned_arms": norm_text(trial.get("Sample size: planned number of arms")),
        "by_arm": norm_text(trial.get("Sample size (or number of clusters) by treatment arms")),
        "mde": norm_text(trial.get("Minimum detectable effect size for main outcomes (accounting for sampledesign and clustering)")),
        # post-trial (if present)
        "final_clusters": norm_text(trial.get("Final Sample Size: Number of Clusters (Unit of Randomization)")),
        "final_observations": norm_text(trial.get("Final Sample Size: Total Number of Observations")),
        "final_by_arm": norm_text(trial.get("Final Sample Size (or Number of Clusters) by Treatment Arms")),
        "attrition_correlated": norm_text(trial.get("Was attrition correlated with treatment status?")),
    }


# Helper functions removed: extract_arms, build_trial_card
# Logic is now handled by the LLM enrichment stage.


def main():
    ap = argparse.ArgumentParser()
    # Simplified paths
    ap.add_argument("--in", dest="inp", default="data/trials_sampled_50.json", help="Input JSON file")
    ap.add_argument("--out", dest="out", default="data/design_specs.jsonl", help="Output JSONL file")
    args = ap.parse_args()

    data_path = Path(args.inp)
    data = json.loads(data_path.read_text(encoding="utf-8"))

    # Support both dict-of-dicts ({"0": {...}, ...}) and list-of-dicts
    if isinstance(data, dict):
        trials = list(data.values())
    elif isinstance(data, list):
        trials = data
    else:
        raise ValueError("Unsupported input structure; expected dict or list")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for t in trials:
            if not isinstance(t, dict):
                continue

            rct_id = norm_text(get_first(t, ["RCT ID", "RCT_ID", "rct_id"])) or "UNKNOWN"
            title = norm_text(t.get("Title"))
            status = norm_text(t.get("Status"))

            # DOI URL: sometimes in "Citation" text; keep best-effort
            citation = norm_text(t.get("Citation"))
            doi_url = ""
            m = re.search(r"(https?://doi\.org/[^\s\"<>]+)", citation, re.IGNORECASE)
            if m:
                doi_url = m.group(1).rstrip(").,]")

            # Countries array often like [{"Country":"Kenya",...}]
            countries = []
            c = t.get("Countries") or t.get("Country names")
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict):
                        name = norm_text(item.get("Country"))
                        if name:
                            countries.append(name)
                    elif isinstance(item, str):
                        countries.append(norm_text(item))
            elif isinstance(c, str):
                countries = [x for x in split_bullets(c) if x]

            spec: Dict[str, Any] = {
                "rct_id": rct_id,
                "title": title,
                "status": status,
                "doi_url": doi_url,
                "countries": countries,

                "start_date": norm_text(t.get("Start date")),
                "end_date": norm_text(t.get("End date")),
                "intervention_start_date": norm_text(t.get("Intervention Start Date")),
                "intervention_end_date": norm_text(t.get("Intervention End Date")),

                "randomization_unit": norm_text(t.get("Randomization Unit")),
                "randomization_method": norm_text(t.get("Randomization Method")),
                # "clustered" is deprecated; strict is_clustered is derived by LLM now.
                # We keep raw registry fields as-is if needed, but for "design spec" we rely on LLM.

                "primary_outcomes": split_bullets(t.get("Primary Outcomes (end points)") or ""),
                "primary_outcomes_explanation": norm_text(t.get("Primary Outcomes (explanation)")),
                "secondary_outcomes": split_bullets(t.get("Secondary Outcomes (end points)") or ""),
                "secondary_outcomes_explanation": norm_text(t.get("Secondary Outcomes (explanation)")),

                "intervention_text": norm_text(t.get("Intervention(s)")),
                "experimental_design": norm_text(t.get("Experimental Design")),
                "experimental_design_details": norm_text(t.get("Experimental Design Details")),

                "sample_sizes": parse_sample_sizes(t),
                "keywords": split_bullets(t.get("Keywords") or ""),
                "additional_keywords": t.get("Additional Keywords") if isinstance(t.get("Additional Keywords"), list) else split_bullets(t.get("Additional Keywords") or ""),
                
                # Fields to be populated by LLM enrichment
                "arms": [],
                "factors": [],
                "design_type": None,
                "is_clustered": None,
                "unit_of_randomization_canonical": None,
                "evidence_quotes": [],
            }

            f.write(json.dumps(spec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} design specs to {out_path}")


if __name__ == "__main__":
    main()
