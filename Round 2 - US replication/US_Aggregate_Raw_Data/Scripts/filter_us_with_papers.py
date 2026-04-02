"""
filter_us_with_papers.py

Step 1 of the aggregate pipeline.

Filters the full AEA registry to US completed studies that have at least one
linked paper (with a URL we can use to fetch the PDF). Produces
Data/expanded_us_pool.jsonl — one record per study, normalised to the field
names expected by downstream scripts.

Run from this script's directory:
    python filter_us_with_papers.py
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]  # llm-simulations/
REGISTRY_PATH = REPO_ROOT / "Base Data - AEA files" / "trials.json"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "Data" / "expanded_us_pool.jsonl"


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def is_us(r: dict) -> bool:
    return "United States" in str(r.get("Countries", ""))


def is_completed(r: dict) -> bool:
    return (
        "complet" in str(r.get("Status", "")).lower()
        or str(r.get("Is the intervention completed?", "")).strip().lower() == "yes"
        or str(r.get("Data Collection Complete", "")).strip().lower() == "yes"
    )


def extract_papers(raw) -> list[dict]:
    """Return list of {url, citation, abstract} dicts from whatever the registry stores."""
    if not raw or str(raw).strip() in ("", "None", "[]", "nan"):
        return []
    if isinstance(raw, list):
        return [
            {
                "url": p.get("URL", "").strip() if isinstance(p, dict) else "",
                "citation": p.get("Citation", "").strip() if isinstance(p, dict) else "",
                "abstract": p.get("Abstract", "").strip() if isinstance(p, dict) else "",
            }
            for p in raw
            if isinstance(p, dict) and p.get("URL", "").strip()
        ]
    # Sometimes stored as a JSON string
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return extract_papers(parsed)
        except Exception:
            pass
    return []


def has_usable_paper(r: dict) -> bool:
    return bool(extract_papers(r.get("Papers")))


def normalise(idx: int, r: dict) -> dict:
    """Produce a clean record matching the field names used by enrich_design_specs_llm."""
    papers = extract_papers(r.get("Papers"))
    return {
        "seq_id": idx,
        "rct_id": r.get("RCT ID", ""),
        "title": r.get("Title", ""),
        "status": r.get("Status", ""),
        "countries": r.get("Countries", ""),
        "start_date": r.get("Start date", ""),
        "end_date": r.get("End date", ""),
        "intervention_text": r.get("Intervention(s)", ""),
        "experimental_design": r.get("Experimental Design", ""),
        "experimental_design_details": r.get("Experimental Design Details", ""),
        "primary_outcomes": r.get("Primary Outcomes (end points)", ""),
        "primary_outcomes_explanation": r.get("Primary Outcomes (explanation)", ""),
        "secondary_outcomes": r.get("Secondary Outcomes (end points)", ""),
        "randomization_unit": r.get("Randomization Unit", ""),
        "randomization_method": r.get("Randomization Method", ""),
        "clustered": r.get("Was the treatment clustered?", ""),
        "sample_sizes": r.get("Sample size (or number of clusters) by treatment arms", ""),
        "keywords": r.get("Keywords", ""),
        "abstract": r.get("Abstract", ""),
        "has_public_data": bool(str(r.get("Public Data URL", "")).strip()
                                not in ("", "None", "nan")),
        "public_data_url": r.get("Public Data URL", ""),
        "program_files_url": r.get("Program Files URL", ""),
        "papers": papers,
        # convenience: first paper URL for quick access
        "primary_paper_url": papers[0]["url"] if papers else "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading registry from {REGISTRY_PATH} ...")
    with open(REGISTRY_PATH) as f:
        raw = json.load(f)

    records = list(raw.values()) if isinstance(raw, dict) else raw
    print(f"Total registry entries: {len(records)}")

    pool = []
    for r in records:
        if not isinstance(r, dict):
            continue
        if is_us(r) and is_completed(r) and has_usable_paper(r):
            pool.append(r)

    print(f"US completed with usable paper URL: {len(pool)}")

    # Assign stable seq_ids (1-indexed, sorted by RCT ID for reproducibility)
    pool.sort(key=lambda r: r.get("RCT ID", ""))
    normalised = [normalise(i + 1, r) for i, r in enumerate(pool)]

    # Break down by data availability
    with_data = sum(1 for r in normalised if r["has_public_data"])
    without_data = len(normalised) - with_data
    print(f"  Of which have public microdata URL : {with_data}  (Track 1 overlap)")
    print(f"  Of which paper-only (net new)       : {without_data}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in normalised:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(normalised)} records → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
