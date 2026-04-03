"""
match_ids_llm.py

Uses GPT-4o-mini to build a mapping from simulation arm/outcome IDs
to ground-truth arm/outcome IDs in study_enriched_aggregate_pass2.jsonl.

Mismatches arise because treatment_variations and outcome_questions in
simulatable_studies.json were extracted from the instrument/methods section
of each paper, while the GT in study_enriched_aggregate.jsonl was extracted
independently from the results section — so the same concepts ended up with
different labels.

The mapping is written to:
    Data/id_mapping.json

And consumed by compare_aggregate.py via --mapping flag.

Usage:
    python match_ids_llm.py           # skips already-cached studies
    python match_ids_llm.py --force   # re-run all studies
    python match_ids_llm.py --dry-run # print prompts without calling API
"""

import argparse, json, random, re, sys, time
from pathlib import Path
from openai import OpenAI, RateLimitError, APIConnectionError, APIError

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR.parents[1] / "Data"
STUDIES_PATH = DATA_DIR / "simulatable_studies.json"
ENRICHED     = DATA_DIR / "Ground_Truth" / "study_enriched_aggregate_pass2.jsonl"
MAPPING_OUT  = DATA_DIR / "id_mapping.json"
CACHE_PATH   = DATA_DIR / ".match_cache.json"

MODEL = "gpt-5.4-mini"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50]


def load_data():
    with open(STUDIES_PATH) as f:
        sims = {s["seq_id"]: s for s in json.load(f)}

    gt_map = {}
    for line in open(ENRICHED):
        r = json.loads(line)
        if r.get("extract_status") == "ok":
            gt_map[r["seq_id"]] = r

    return sims, gt_map


def build_arm_tables(sim: dict, gt: dict):
    """
    Returns:
        sim_arms: {arm_id: {"label": ..., "text": ...}}   (from treatment_variations)
        gt_arms:  {arm_id: {"label": ...}}                (from group_summaries)
    """
    sim_arms = {}
    for v in sim.get("treatment_variations", []):
        arm_id = v.get("arm_id") or slugify(v.get("arm_label", ""))
        sim_arms[arm_id] = {
            "label": v.get("arm_label", arm_id),
            "text":  v.get("text", "")[:300],     # truncate for token budget
        }

    gt_arms = {}
    for outcome in (gt.get("results") or {}).get("outcomes", []):
        for gs in outcome.get("group_summaries", []):
            arm_id = gs.get("arm_id")
            if arm_id and arm_id not in gt_arms:
                gt_arms[arm_id] = {"label": gs.get("arm_label", arm_id)}

    return sim_arms, gt_arms


def build_outcome_tables(sim: dict, gt: dict):
    """
    Returns:
        sim_outs: {outcome_id: {"name": ..., "question": ...}}
        gt_outs:  {outcome_id: {"name": ...}}
    """
    sim_outs = {}
    for q in sim.get("outcome_questions", []):
        oid = slugify(q.get("outcome_name", ""))
        sim_outs[oid] = {
            "name":     q.get("outcome_name", oid),
            "question": q.get("question_text", "")[:200],
        }

    gt_outs = {}
    for outcome in (gt.get("results") or {}).get("outcomes", []):
        name = outcome.get("name", "")
        oid  = slugify(name)
        gt_outs[oid] = {"name": name}

    return sim_outs, gt_outs


# ---------------------------------------------------------------------------
# LLM matching
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are a careful research assistant reconciling variable names "
    "across two extractions of the same social science experiment. "
    "Your job is to match simulation labels to ground-truth labels "
    "from the published paper. Respond only with the requested JSON."
)

def build_prompt(seq_id: int, title: str,
                 sim_arms, gt_arms,
                 sim_outs, gt_outs) -> str:

    arms_already_match  = [k for k in sim_arms if k in gt_arms]
    arms_need_matching  = {k: v for k, v in sim_arms.items() if k not in gt_arms}
    outs_already_match  = [k for k in sim_outs if k in gt_outs]
    outs_need_matching  = {k: v for k, v in sim_outs.items() if k not in gt_outs}

    return f"""Study: "{title}"  (seq_id={seq_id})

--- ARMS ---

Already matched (exact ID): {arms_already_match}

Simulation arms needing a match:
{json.dumps(arms_need_matching, indent=2)}

Available ground-truth arm IDs (from paper results section):
{json.dumps({k: v['label'] for k, v in gt_arms.items()}, indent=2)}

--- OUTCOMES ---

Already matched (exact ID): {outs_already_match}

Simulation outcomes needing a match:
{json.dumps(outs_need_matching, indent=2)}

Available ground-truth outcome IDs (from paper results section):
{json.dumps({k: v['name'] for k, v in gt_outs.items()}, indent=2)}

--- INSTRUCTIONS ---

For each unmatched simulation arm/outcome, find the best ground-truth ID.
Rules:
  • Match when the names clearly refer to the same concept, even if worded differently.
  • Multiple simulation IDs may map to the same GT ID (e.g., factorial cells
    collapsing to a marginal condition).
  • Use null if there is NO reasonable match (e.g., attention/comprehension
    checks with no GT counterpart, or completely different constructs).
  • Do NOT invent GT IDs — only use IDs from the "Available" lists above.

Respond with ONLY valid JSON:
{{
  "arms":     {{ "<sim_arm_id>": "<gt_arm_id or null>", ... }},
  "outcomes": {{ "<sim_outcome_id>": "<gt_outcome_id or null>", ... }}
}}
"""


def call_llm(client: OpenAI, prompt: str, dry_run: bool,
             retries: int = 6) -> dict | None:
    if dry_run:
        print("── DRY RUN PROMPT ──")
        print(prompt)
        print()
        return None

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",  "content": SYSTEM_MSG},
                    {"role": "user",    "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
            return json.loads(raw)
        except RateLimitError as e:
            wait = min(120, 20 * 2 ** attempt) + random.uniform(0, 5)
            print(f"  Rate limited (attempt {attempt+1}/{retries}), "
                  f"retrying in {wait:.0f}s … ({e})")
            time.sleep(wait)
        except APIConnectionError as e:
            wait = min(30, 5 * 2 ** attempt) + random.uniform(0, 2)
            print(f"  Connection error (attempt {attempt+1}/{retries}), "
                  f"retrying in {wait:.0f}s … ({e})")
            time.sleep(wait)
        except APIError as e:
            wait = min(60, 10 * 2 ** attempt) + random.uniform(0, 3)
            print(f"  API error (attempt {attempt+1}/{retries}), "
                  f"retrying in {wait:.0f}s … ({e})")
            time.sleep(wait)
        except json.JSONDecodeError as e:
            print(f"  JSON decode error (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return None
    print(f"  Giving up after {retries} attempts.")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force",   action="store_true",
                        help="Re-run all studies even if cached")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling the API")
    args = parser.parse_args()

    sims, gt_map = load_data()

    # Load cache
    try:
        cache: dict = json.loads(CACHE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    # Load existing mapping (if any) as base
    try:
        mapping: dict = json.loads(MAPPING_OUT.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        mapping = {}

    client = None if args.dry_run else OpenAI()

    stats = {"processed": 0, "cached": 0, "skipped_no_gt": 0,
             "arms_mapped": 0, "outcomes_mapped": 0}

    for seq_id in sorted(sims):
        sim = sims[seq_id]
        gt  = gt_map.get(seq_id)
        title = (gt or {}).get("title", f"seq={seq_id}")

        if gt is None:
            print(f"seq={seq_id:>3}  SKIP  (no GT extraction)")
            stats["skipped_no_gt"] += 1
            continue

        sim_arms, gt_arms = build_arm_tables(sim, gt)
        sim_outs, gt_outs = build_outcome_tables(sim, gt)

        arms_need  = [k for k in sim_arms if k not in gt_arms]
        outs_need  = [k for k in sim_outs if k not in gt_outs]

        if not arms_need and not outs_need:
            print(f"seq={seq_id:>3}  EXACT MATCH  {title[:55]}")
            # Ensure a pass-through entry exists in mapping
            mapping.setdefault(str(seq_id), {"arms": {}, "outcomes": {}})
            continue

        cache_key = str(seq_id)
        if not args.force and cache_key in cache:
            print(f"seq={seq_id:>3}  CACHED  {title[:55]}")
            result = cache[cache_key]
            stats["cached"] += 1
        else:
            prompt = build_prompt(seq_id, title,
                                  sim_arms, gt_arms,
                                  sim_outs, gt_outs)
            print(f"seq={seq_id:>3}  CALLING API  {title[:55]}")
            result = call_llm(client, prompt, args.dry_run)
            if result is None:
                continue
            cache[cache_key] = result
            CACHE_PATH.write_text(json.dumps(cache, indent=2))
            stats["processed"] += 1

        # Merge into mapping (only mismatched ids)
        study_map = mapping.setdefault(str(seq_id), {"arms": {}, "outcomes": {}})
        for sim_arm, gt_arm in (result.get("arms") or {}).items():
            if sim_arm not in gt_arms:          # only store if not already exact
                study_map["arms"][sim_arm] = gt_arm
                if gt_arm:
                    stats["arms_mapped"] += 1
        for sim_out, gt_out in (result.get("outcomes") or {}).items():
            if sim_out not in gt_outs:
                study_map["outcomes"][sim_out] = gt_out
                if gt_out:
                    stats["outcomes_mapped"] += 1

        # Report what was mapped
        mapped_arms = {k: v for k, v in study_map["arms"].items() if v}
        null_arms   = [k for k, v in study_map["arms"].items() if not v]
        mapped_outs = {k: v for k, v in study_map["outcomes"].items() if v}
        null_outs   = [k for k, v in study_map["outcomes"].items() if not v]
        if mapped_arms:
            print(f"        arms mapped  : {mapped_arms}")
        if null_arms:
            print(f"        arms → null  : {null_arms}")
        if mapped_outs:
            print(f"        outs mapped  : {mapped_outs}")
        if null_outs:
            print(f"        outs → null  : {null_outs}")

    # Write final mapping
    if not args.dry_run:
        MAPPING_OUT.write_text(json.dumps(mapping, indent=2))
        print(f"\nMapping written → {MAPPING_OUT}")

    print(f"\n── Summary ──")
    print(f"  Processed via API : {stats['processed']}")
    print(f"  Loaded from cache : {stats['cached']}")
    print(f"  Skipped (no GT)   : {stats['skipped_no_gt']}")
    print(f"  Arms mapped       : {stats['arms_mapped']}")
    print(f"  Outcomes mapped   : {stats['outcomes_mapped']}")
    print()
    print("Next step:")
    print("  python compare_aggregate.py --config no_reasoning --mapping")


if __name__ == "__main__":
    main()
