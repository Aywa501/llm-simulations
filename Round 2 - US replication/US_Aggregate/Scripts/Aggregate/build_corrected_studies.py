"""
build_corrected_studies.py

Builds simulatable_studies_v2.json with maximum arm coverage by:

  1. Applying id_mapping.json (from match_ids_llm.py) to fix naming
     mismatches between simulation arm/outcome IDs and GT IDs.

  2. For arms that STILL lack GT values after mapping, re-extracting
     results directly from the paper PDFs in Data/Papers/.  This catches
     cases where the original extraction missed arms the paper does report.

  3. Writing newly extracted values back into
     Data/Ground_Truth/study_enriched_aggregate_pass2.jsonl.

The final simulatable_studies_v2.json contains every arm that has (or now
gets) GT coverage.  Arms are only dropped when the paper genuinely does not
report at that granularity (the LLM returns found=false).

Usage:
    python build_corrected_studies.py [--dry-run] [--force]

    --dry-run  show what would change; do not write any files
    --force    re-run PDF extraction even for cached studies

Prerequisites:
    id_mapping.json   — run match_ids_llm.py first
"""

import argparse, json, random, re, time
from pathlib import Path

import pdfplumber
from openai import OpenAI, RateLimitError, APIConnectionError, APIError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR.parents[1] / "Data"
STUDIES_IN   = DATA_DIR / "simulatable_studies.json"
MAPPING_PATH = DATA_DIR / "id_mapping.json"
GT_PATH      = DATA_DIR / "Ground_Truth" / "study_enriched_aggregate_pass2.jsonl"
PAPERS_ROOT  = DATA_DIR / "Papers"
STUDIES_OUT  = DATA_DIR / "simulatable_studies_v2.json"
EXTRACT_CACHE = DATA_DIR / ".build_corrected_cache.json"

EXTRACT_MODEL = "gpt-5.1"        # heavy extraction from PDFs
MAX_PDF_CHARS = 80_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50]


def find_pdfs(seq_id: int) -> list[Path]:
    """Return all non-supplement PDFs for a study, sorted."""
    pdfs = []
    for person_dir in PAPERS_ROOT.iterdir():
        if not person_dir.is_dir():
            continue
        for p in person_dir.iterdir():
            stem = p.stem
            if "supplement" in stem.lower():
                continue
            # match {seq_id}.pdf or {seq_id}_{n}.pdf
            if re.fullmatch(rf"{seq_id}(_\d+)?", stem):
                pdfs.append(p)
    return sorted(pdfs)


def extract_pdf_text(pdfs: list[Path]) -> str:
    parts = []
    for pdf in pdfs:
        try:
            with pdfplumber.open(pdf) as doc:
                for page in doc.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
        except Exception as e:
            print(f"    WARNING: could not read {pdf.name}: {e}")
    text = "\n\n".join(parts)
    return text[:MAX_PDF_CHARS]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_gt() -> dict[int, dict]:
    gt = {}
    for line in open(GT_PATH):
        r = json.loads(line)
        gt[r["seq_id"]] = r
    return gt


def load_cache() -> dict:
    try:
        return json.loads(EXTRACT_CACHE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict):
    EXTRACT_CACHE.write_text(json.dumps(cache, indent=2))


def gt_arm_ids(gt_record: dict) -> set[str]:
    ids = set()
    for o in (gt_record.get("results") or {}).get("outcomes", []):
        for gs in o.get("group_summaries", []):
            if gs.get("arm_id"):
                ids.add(gs["arm_id"])
    return ids


def gt_outcome_ids(gt_record: dict) -> set[str]:
    return {slugify(o["name"])
            for o in (gt_record.get("results") or {}).get("outcomes", [])
            if o.get("name")}


# ---------------------------------------------------------------------------
# Targeted PDF extraction
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "You are a careful research assistant extracting numerical results "
    "from social science experiment papers. "
    "Always respond with valid JSON only."
)

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "arms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "arm_id":    {"type": "string"},
                    "arm_label": {"type": "string"},
                    "found":     {"type": "boolean"},
                    "outcomes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "outcome_name": {"type": "string"},
                                "n_analyzed":   {"type": ["number", "null"]},
                                "metric":       {"type": "string"},
                                "value":        {"type": ["number", "null"]},
                                "sd":           {"type": ["number", "null"]},
                                "se":           {"type": ["number", "null"]},
                                "evidence":     {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
    },
}


def build_extraction_prompt(
    title: str,
    existing_arm_labels: list[str],
    missing_arms: list[dict],      # [{arm_id, arm_label, text}]
    outcome_questions: list[dict], # [{outcome_name, question_text, response_format}]
    pdf_text: str,
) -> str:

    arms_block = json.dumps(
        [{"arm_id": a["arm_id"],
          "arm_label": a["arm_label"],
          "treatment_text": a.get("text", "")[:400]}
         for a in missing_arms],
        indent=2,
    )
    outcomes_block = json.dumps(
        [{"outcome_name": q.get("outcome_name", ""),
          "question":     q.get("question_text", q.get("response_instruction", ""))[:200]}
         for q in outcome_questions],
        indent=2,
    )

    return f"""You are extracting missing experimental results from a published paper.

Study title: "{title}"

The following arms were already found in the paper:
{existing_arm_labels}

We need results for these MISSING arms (they are real treatment conditions from the study):
{arms_block}

We are interested in these outcome measures:
{outcomes_block}

--- PAPER TEXT (first {len(pdf_text):,} chars) ---
{pdf_text}
--- END PAPER TEXT ---

For each missing arm, search the paper and report:
- found: true if the paper reports any numerical result for this arm, false otherwise
- outcomes: for each outcome, the reported mean/proportion, N, SD/SE (null if not reported)
- evidence: a short verbatim quote from the paper confirming the value

If the paper does not report results at this arm granularity (e.g., results are only
reported as aggregate comparisons), set found=false.

Respond with JSON matching this structure:
{{
  "arms": [
    {{
      "arm_id": "<same as input>",
      "arm_label": "<same as input>",
      "found": true,
      "outcomes": [
        {{
          "outcome_name": "<outcome name>",
          "n_analyzed": 500,
          "metric": "mean",
          "value": 3.2,
          "sd": 1.1,
          "se": null,
          "evidence": "Table 2 shows group G2 mean = 3.2 (SD=1.1)"
        }}
      ]
    }}
  ]
}}
"""


def call_extraction(client: OpenAI, prompt: str, retries: int = 6) -> dict | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=EXTRACT_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except RateLimitError as e:
            wait = min(120, 20 * 2 ** attempt) + random.uniform(0, 5)
            print(f"    Rate limited (attempt {attempt+1}/{retries}), "
                  f"retrying in {wait:.0f}s … ({e})")
            time.sleep(wait)
        except APIConnectionError as e:
            wait = min(30, 5 * 2 ** attempt) + random.uniform(0, 2)
            print(f"    Connection error (attempt {attempt+1}/{retries}), "
                  f"retrying in {wait:.0f}s … ({e})")
            time.sleep(wait)
        except APIError as e:
            wait = min(60, 10 * 2 ** attempt) + random.uniform(0, 3)
            print(f"    API error (attempt {attempt+1}/{retries}), "
                  f"retrying in {wait:.0f}s … ({e})")
            time.sleep(wait)
        except json.JSONDecodeError as e:
            print(f"    JSON decode error (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return None
    print(f"    Giving up after {retries} attempts.")
    return None


# ---------------------------------------------------------------------------
# Merge extracted arm values into GT record
# ---------------------------------------------------------------------------

def merge_into_gt(gt_record: dict, extracted_arms: list[dict]) -> int:
    """
    Adds newly extracted arm group_summaries into gt_record in-place.
    Returns the number of new arm×outcome values added.
    """
    results = gt_record.setdefault("results", {"outcomes": []})
    outcomes_list = results.setdefault("outcomes", [])

    # Build index: outcome_name_slug → outcome entry
    out_index: dict[str, dict] = {}
    for o in outcomes_list:
        out_index[slugify(o.get("name", ""))] = o

    added = 0
    for arm_data in extracted_arms:
        if not arm_data.get("found"):
            continue
        arm_id    = arm_data["arm_id"]
        arm_label = arm_data.get("arm_label", arm_id)

        for out_data in arm_data.get("outcomes", []):
            value = out_data.get("value")
            if value is None:
                continue

            out_name = out_data.get("outcome_name", "")
            out_slug = slugify(out_name)

            # Find or create the outcome entry
            if out_slug not in out_index:
                new_out = {"name": out_name, "is_primary": False,
                           "outcome_type": "continuous", "group_summaries": []}
                outcomes_list.append(new_out)
                out_index[out_slug] = new_out

            outcome_entry = out_index[out_slug]
            gs_list = outcome_entry.setdefault("group_summaries", [])

            # Don't overwrite an existing entry for this arm
            if any(gs.get("arm_id") == arm_id for gs in gs_list):
                continue

            gs_list.append({
                "arm_label":  arm_label,
                "arm_id":     arm_id,
                "n_analyzed": out_data.get("n_analyzed"),
                "metric":     out_data.get("metric", "mean"),
                "value":      value,
                "sd":         out_data.get("sd"),
                "se":         out_data.get("se"),
                "evidence":   out_data.get("evidence", ""),
                "source":     "build_corrected_studies",
            })
            added += 1

    return added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force",   action="store_true",
                        help="Re-run PDF extraction even if cached")
    args = parser.parse_args()

    # --- Load inputs ---
    with open(STUDIES_IN) as f:
        studies = json.load(f)

    if not MAPPING_PATH.exists():
        print(f"ERROR: {MAPPING_PATH.name} not found. Run match_ids_llm.py first.")
        raise SystemExit(1)

    mapping: dict = json.loads(MAPPING_PATH.read_text())
    gt_data: dict[int, dict] = load_gt()
    cache: dict = load_cache()

    client = None if args.dry_run else OpenAI()

    out_studies   = []
    gt_updated    = set()    # seq_ids whose GT was augmented
    total_added   = 0

    for sim in studies:
        seq_id    = sim["seq_id"]
        study_map = mapping.get(str(seq_id), {})
        arm_map   = study_map.get("arms",     {})
        out_map   = study_map.get("outcomes", {})
        gt_rec    = gt_data.get(seq_id, {})
        existing_arm_ids  = gt_arm_ids(gt_rec)
        existing_out_ids  = gt_outcome_ids(gt_rec)
        title = gt_rec.get("title", f"seq={seq_id}")

        # ── Step 1: compute canonical arm_id for each sim arm ──
        # canonical = GT arm_id if known via mapping; else keep sim arm_id
        # (newly extracted arms will carry the sim arm_id as their GT arm_id)
        resolved_variations = []
        arms_needing_extraction = []   # [{arm_id, arm_label, text}]
        seen_canonical: set[str] = set()

        for v in sim.get("treatment_variations", []):
            orig_id = v.get("arm_id") or slugify(v.get("arm_label", ""))

            # Resolve via mapping
            if orig_id in arm_map:
                canonical = arm_map[orig_id]      # may be None
            else:
                canonical = orig_id               # exact match or new ID

            if canonical is None:
                # Mapping said no GT equivalent — still try extraction with orig_id
                canonical = orig_id

            # Deduplicate (many-to-one collapses)
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)

            resolved_variations.append({**v, "arm_id": canonical,
                                         "arm_label": v.get("arm_label", canonical)})

            if canonical not in existing_arm_ids:
                arms_needing_extraction.append({
                    "arm_id":    canonical,
                    "arm_label": v.get("arm_label", canonical),
                    "text":      v.get("text", ""),
                })

        # ── Step 2: compute canonical outcome for each sim outcome ──
        resolved_outcomes = []
        outs_needing_extraction = []

        for q in sim.get("outcome_questions", []):
            orig_id = slugify(q.get("outcome_name", ""))

            if orig_id in out_map:
                canonical = out_map[orig_id]
            else:
                canonical = orig_id

            if canonical is None:
                canonical = orig_id

            # Use GT outcome name if we have it, else keep original
            canonical_name = q.get("outcome_name", canonical)
            for o in (gt_rec.get("results") or {}).get("outcomes", []):
                if slugify(o.get("name", "")) == canonical:
                    canonical_name = o["name"]
                    break

            resolved_outcomes.append({**q, "outcome_name": canonical_name})

            if canonical not in existing_out_ids:
                outs_needing_extraction.append(canonical_name)

        # ── Step 3: targeted PDF extraction for missing arms ──
        if arms_needing_extraction:
            pdfs = find_pdfs(seq_id)
            if not pdfs:
                print(f"seq={seq_id:>3}  no PDF found — keeping {len(resolved_variations)} arms, "
                      f"{len(arms_needing_extraction)} without GT")
            else:
                cache_key = f"{seq_id}__{'_'.join(a['arm_id'] for a in arms_needing_extraction)}"

                if not args.force and cache_key in cache:
                    print(f"seq={seq_id:>3}  extraction cached  ({len(arms_needing_extraction)} missing arms)")
                    extracted_arms = cache[cache_key]
                else:
                    print(f"seq={seq_id:>3}  extracting {len(arms_needing_extraction)} missing arms "
                          f"from {[p.name for p in pdfs]}  ...")
                    if args.dry_run:
                        extracted_arms = []
                    else:
                        pdf_text = extract_pdf_text(pdfs)
                        existing_labels = [
                            gs.get("arm_label", gs.get("arm_id", ""))
                            for o in (gt_rec.get("results") or {}).get("outcomes", [])
                            for gs in o.get("group_summaries", [])
                        ]
                        prompt = build_extraction_prompt(
                            title,
                            existing_labels,
                            arms_needing_extraction,
                            sim.get("outcome_questions", []),
                            pdf_text,
                        )
                        result = call_extraction(client, prompt)
                        extracted_arms = (result or {}).get("arms", [])
                        cache[cache_key] = extracted_arms
                        save_cache(cache)

                if not args.dry_run and extracted_arms:
                    n = merge_into_gt(gt_rec, extracted_arms)
                    if n:
                        total_added += n
                        gt_updated.add(seq_id)
                        print(f"        → added {n} new arm×outcome values to GT")

                    # Report what was found vs not found
                    found    = [a["arm_id"] for a in extracted_arms if a.get("found")]
                    not_fnd  = [a["arm_id"] for a in extracted_arms if not a.get("found")]
                    if found:   print(f"        found    : {found}")
                    if not_fnd: print(f"        not found: {not_fnd}")

                    # Remove arms the paper truly does not report
                    not_found_ids = set(not_fnd)
                    resolved_variations = [
                        v for v in resolved_variations
                        if v["arm_id"] not in not_found_ids
                        or v["arm_id"] in existing_arm_ids
                    ]

        # ── Step 4: report and collect ──
        n_arms_orig = len(sim.get("treatment_variations", []))
        n_outs_orig = len(sim.get("outcome_questions", []))

        print(f"seq={seq_id:>3}  arms {n_arms_orig}→{len(resolved_variations)}  "
              f"outs {n_outs_orig}→{len(resolved_outcomes)}  "
              f"({title[:45]})")

        if not resolved_variations or not resolved_outcomes:
            print(f"        DROPPED (nothing survived)")
            continue

        out_studies.append({
            **sim,
            "treatment_variations": resolved_variations,
            "outcome_questions":    resolved_outcomes,
        })

    # ── Write outputs ──
    print(f"\n── Summary ──")
    print(f"  Studies in v2      : {len(out_studies)}")
    print(f"  GT records updated : {len(gt_updated)}  (seq_ids: {sorted(gt_updated)})")
    print(f"  New GT values added: {total_added}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # Write corrected study configs
    with open(STUDIES_OUT, "w") as f:
        json.dump(out_studies, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {STUDIES_OUT.name}  ({len(out_studies)} studies)")

    # Write updated GT (all records, updated in-place for modified ones)
    with open(GT_PATH, "w") as f:
        for rec in gt_data.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {GT_PATH.name}  ({len(gt_data)} records, {len(gt_updated)} updated)")

    print("\nNext steps:")
    print("  python simulate_aggregate.py --generate-only "
          "--studies-file simulatable_studies_v2.json --config no_reasoning --n 50")


if __name__ == "__main__":
    main()
