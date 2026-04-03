"""
extract_from_paper.py

Step 4 of the aggregate pipeline.

For each Tier 1 study with a downloaded PDF, extracts two things via GPT:
  (A) Aggregate results  — per-arm N, outcome name, metric, value, SD/SE
  (B) Instrument text    — the actual survey question / experimental stimulus
                           shown to participants, including per-arm variations
                           and the exact question wording + response format

PDFs are located in Data/Papers/{person}/ with naming convention:
  {seq_id}.pdf          — single paper
  {seq_id}_1.pdf etc.   — multiple papers for one study (all merged)
  {seq_id}_supplement.pdf — excluded

PDF text is extracted with pdfplumber before being sent to the model.

Outputs Data/study_enriched_aggregate.jsonl, one record per study.
Also maintains a .extract_cache.json to avoid re-running completed studies.

Requires:
  - OPENAI_API_KEY in environment
  - pip install pdfplumber
  - Data/tier_classified.jsonl      (from classify_tiers_llm.py)
  - Data/Papers/{person}/{seq_id}.pdf

Run from project root or this script's directory:
    python extract_from_paper.py [--limit N] [--model MODEL] [--seq SEQ_ID]
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import pdfplumber
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR.parent / "Data"
INPUT       = DATA_DIR / "Pipeline" / "tier_classified.jsonl"
PAPERS_ROOT = DATA_DIR / "Papers"
OUTPUT      = DATA_DIR / "Ground_Truth" / "study_enriched_aggregate.jsonl"
OUTPUT_PASS2 = DATA_DIR / "Ground_Truth" / "study_enriched_aggregate_pass2.jsonl"
CACHE_FILE  = DATA_DIR / "Caches" / ".extract_cache.json"

DEFAULT_MODEL  = "gpt-5.4"
MAX_CONCURRENT = 5
MAX_PDF_CHARS  = 80_000   # chars per study (merged across papers)

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def find_pdfs_for_study(seq_id: int) -> list[Path]:
    """
    Find all primary PDFs for a study anywhere within Papers/.
    Matches: {seq_id}.pdf  and  {seq_id}_{N}.pdf
    Excludes: {seq_id}_supplement.pdf
    """
    found = []
    if not PAPERS_ROOT.exists():
        return found
    for f in PAPERS_ROOT.rglob("*.pdf"):
        stem = f.stem
        if stem == str(seq_id):
            found.append(f)
        elif re.match(rf"^{seq_id}_\d+$", stem):
            found.append(f)
    return sorted(found)


def extract_pdf_text(paths: list[Path]) -> str:
    """Extract and merge text from one or more PDFs, up to MAX_PDF_CHARS."""
    parts = []
    for path in paths:
        pages = []
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
        except Exception as e:
            pages.append(f"[PDF extraction failed for {path.name}: {e}]")
        if pages:
            parts.append(f"=== {path.name} ===\n" + "\n\n".join(pages))

    merged = "\n\n".join(parts)
    if len(merged) > MAX_PDF_CHARS:
        merged = merged[:MAX_PDF_CHARS] + "\n[... truncated ...]"
    return merged


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------

RESULTS_SCHEMA_DESC = """
{
  "outcomes": [
    {
      "name": "<outcome name>",
      "is_primary": true | false,
      "outcome_type": "binary" | "continuous" | "count" | "ordinal" | "categorical" | "other",
      "table_reference": "<e.g. Table 2, col 3 — or null>",
      "group_summaries": [
        {
          "arm_label": "<arm name exactly as written in paper>",
          "n_analyzed": <integer or null>,
          "metric": "mean" | "proportion" | "rate" | "count" | "index" | "other",
          "value": <number or null>,
          "sd": <number or null>,
          "se": <number or null>,
          "ci_lower": <number or null>,
          "ci_upper": <number or null>
        }
      ]
    }
  ],
  "extraction_notes": "<caveats, ambiguities, or assumptions made>"
}"""

INSTRUMENT_SCHEMA_DESC = """
{
  "found": true | false,
  "preamble": "<shared context shown to ALL participants before any arm-specific content — verbatim or near-verbatim, or null>",
  "treatment_variations": [
    {
      "arm_label": "<arm name exactly as in paper>",
      "text": "<arm-specific stimulus: the text, vignette, or information that distinguishes this arm>"
    }
  ],
  "outcome_questions": [
    {
      "outcome_name": "<name matching one of the outcomes in the results extraction>",
      "question_text": "<exact wording of the survey question or task instruction shown to participants>",
      "response_format": "binary" | "percent" | "scale" | "dollar" | "choice" | "integer" | "proportion" | "other",
      "scale_min": <number or null>,
      "scale_max": <number or null>,
      "response_instruction": "<single sentence telling the LLM exactly how to answer, e.g. 'Reply YES or NO only.' or 'Reply with a single integer from 0 to 100 only.'>"
    }
  ],
  ],
  "source_location": "<e.g. Appendix A, p.8 — or null>"
}"""

SECOND_PASS_SCHEMA_DESC = """
{
  "outcomes": [
    {
      "name": "<outcome name>",
      "group_summaries": [
        {
          "arm_label": "<arm name exactly as written in paper>",
          "n_analyzed": <integer or null>,
          "sd": <number or null>,
          "se": <number or null>
        }
      ]
    }
  ]
}"""

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RESULTS_SYSTEM = (
    "You are extracting experimental results from a social science paper. "
    "Extract the results for the primary outcome(s) across ALL experimental arms (all treatment variations and controls) reported in the paper. "
    "Be entirely comprehensive: include every condition that the paper explicitly tests and reports data for. "
    "Report arm labels and numbers exactly as they appear in the paper. "
    "Use null for any value not reported. "
    "Respond with valid JSON only — no markdown fences, no explanation outside JSON."
)

INSTRUMENT_SYSTEM = (
    "You are extracting the experimental stimulus and survey instrument from a social "
    "science paper. Look in the main text, footnotes, and appendices. "
    "You will be given a definitive, anchored list of experimental group labels extracted from the results section. "
    "You must find the verbatim text (survey questions, vignettes, informational treatments) shown to participants for EXACTLY these arms, and no others. "
    "PREAMBLE: shared text shown to all arms before any manipulation. "
    "TREATMENT_VARIATIONS: the arm-specific text that differs across conditions. Only include the arms listed in the prompt. "
    "OUTCOME_QUESTIONS: the exact question(s) asked of participants after the treatment, "
    "including the precise response scale and a concise instruction for an LLM respondent. "
    "If no instrument text appears in the paper, set found=false. "
    "Respond with valid JSON only — no markdown fences, no explanation outside JSON."
)

SECOND_PASS_SYSTEM = (
    "You are extracting missing sample sizes (n) and standard deviations (SD) from a social science paper. "
    "You will be given a list of outcomes and experimental arms that were previously extracted, but are missing 'n' or 'sd'. "
    "Your job is to read the paper text (especially the tables and methods sections) and find the exact sample size (n) and standard deviation (sd) or standard error (se) for those outcomes and arms. "
    "Respond with valid JSON only — no markdown fences, no explanation outside JSON."
)


def make_results_prompt(rec: dict, pdf_text: str) -> str:
    return (
        f"STUDY: {rec.get('title', '')}\n\n"
        f"INTERVENTION: {str(rec.get('intervention_text', ''))[:600]}\n\n"
        f"EXPERIMENTAL DESIGN: {str(rec.get('experimental_design', ''))[:600]}\n\n"
        f"PRIMARY OUTCOMES: {str(rec.get('primary_outcomes', ''))[:400]}\n\n"
        f"--- PAPER TEXT ---\n{pdf_text}\n--- END PAPER ---\n\n"
        f"Extract the main results following this JSON schema:\n{RESULTS_SCHEMA_DESC}"
    )


def make_instrument_prompt(rec: dict, pdf_text: str, results: dict) -> str:
    anchored_arms = []
    if results and results.get("outcomes"):
        for out in results["outcomes"]:
            for gs in out.get("group_summaries", []):
                anchored_arms.append(gs.get("arm_label"))
    anchored_arms = list(dict.fromkeys(anchored_arms)) # deduplicate
    
    return (
        f"STUDY: {rec.get('title', '')}\n\n"
        f"INTERVENTION: {str(rec.get('intervention_text', ''))[:600]}\n\n"
        f"EXPERIMENTAL DESIGN: {str(rec.get('experimental_design', ''))[:600]}\n\n"
        f"PRIMARY OUTCOMES: {str(rec.get('primary_outcomes', ''))[:400]}\n\n"
        f"ANCHORED ARMS LIST (Extract Text ONLY for These!):\n{json.dumps(anchored_arms, indent=2)}\n\n"
        f"--- PAPER TEXT ---\n{pdf_text}\n--- END PAPER ---\n\n"
        f"Extract the experimental instrument for EXACTLY the anchored arms following this JSON schema:\n{INSTRUMENT_SCHEMA_DESC}"
    )


def make_second_pass_prompt(rec: dict, pdf_text: str) -> str:
    missing_data = []
    if rec.get('results') and rec['results'].get('outcomes'):
        for outcome in rec['results']['outcomes']:
            o_name = outcome.get('name')
            gs_list = []
            for gs in outcome.get('group_summaries', []):
                gs_list.append({
                    "arm_label": gs.get('arm_label'),
                    "current_n": gs.get('n_analyzed'),
                    "current_value": gs.get('value'),
                    "current_sd": gs.get('sd')
                })
            missing_data.append({"outcome": o_name, "arms": gs_list})
            
    return (
        f"STUDY: {rec.get('title', '')}\n\n"
        f"TARGET OUTCOMES AND ARMS:\n{json.dumps(missing_data, indent=2)}\n\n"
        f"--- PAPER TEXT ---\n{pdf_text}\n--- END PAPER ---\n\n"
        f"Fill in the missing n_analyzed, sd, and se values for the target outcomes. "
        f"Use this JSON schema:\n{SECOND_PASS_SCHEMA_DESC}"
    )


# ---------------------------------------------------------------------------
# Arm-ID normalisation
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    """Convert an arm label to a clean snake_case ID, max 50 chars."""
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:50]


def assign_arm_ids(instrument: dict) -> dict:
    """
    Add 'arm_id' to each treatment_variation in-place.
    Guarantees uniqueness with a numeric suffix if slugs collide.
    """
    seen: dict[str, int] = {}
    for v in instrument.get("treatment_variations", []):
        base = slugify(v.get("arm_label", "arm"))
        if base in seen:
            seen[base] += 1
            v["arm_id"] = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            v["arm_id"] = base
    return instrument


def match_result_arm_ids(results: dict, instrument: dict) -> dict:
    """
    Add 'arm_id' to each group_summary by fuzzy-matching arm_label
    against the treatment_variations in the instrument.
    Falls back to slugify(arm_label) if no match found.
    """
    # Build label → arm_id map from instrument
    label_map: dict[str, str] = {}
    for v in instrument.get("treatment_variations", []):
        if "arm_id" in v:
            label_map[v["arm_label"].lower().strip()] = v["arm_id"]

    for outcome in results.get("outcomes", []):
        for gs in outcome.get("group_summaries", []):
            raw_label = gs.get("arm_label", "")
            matched = label_map.get(raw_label.lower().strip())
            gs["arm_id"] = matched if matched else slugify(raw_label)
    return results


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def parse_json_response(text: str) -> dict:
    text = text.strip()
    # strip markdown fences if model added them
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

async def extract_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    rec: dict,
    cache: dict,
    model: str,
) -> dict:
    seq_id = rec["seq_id"]
    pdfs   = find_pdfs_for_study(seq_id)

    if not pdfs:
        return {**rec, "extract_status": "no_pdf", "results": None, "instrument": None,
                "pdf_files": []}

    pdf_text = extract_pdf_text(pdfs)

    results_key    = f"{seq_id}:results"
    instrument_key = f"{seq_id}:instrument"

    print(f"Starting or checking cache for seq_id {seq_id}...")

    async def call_gpt(system: str, user: str) -> dict:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                async with sem:
                    response = await client.chat.completions.create(
                        model=model,
                        temperature=0,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                        ],
                    )
                return parse_json_response(response.choices[0].message.content)
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "rate limit" in err_str or "429" in err_str:
                    wait_time = min(2 ** attempt + 3, 120)
                    print(f"Rate limit hit for seq_id {seq_id}, attempt {attempt+1}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
        raise Exception("Max retries reached")

    # Results extraction
    results = cache.get(results_key)
    if results and ("rate_limit" in str(results).lower() or "rate limit" in str(results).lower() or "429" in str(results)):
        results = None

    if results:
        pass
    else:
        try:
            print(f"Extracting results for seq_id {seq_id}...")
            results = await call_gpt(RESULTS_SYSTEM, make_results_prompt(rec, pdf_text))
            cache[results_key] = results
        except Exception as e:
            results = {"outcomes": [], "extraction_notes": f"failed: {e}"}

    # Instrument extraction
    instrument = cache.get(instrument_key)
    if instrument and ("rate_limit" in str(instrument).lower() or "rate limit" in str(instrument).lower() or "429" in str(instrument)):
        instrument = None

    if instrument:
        pass
    else:
        try:
            print(f"Extracting instrument for seq_id {seq_id}...")
            # Re-fetch from cache, wait if None, or create
            instrument = await call_gpt(INSTRUMENT_SYSTEM, make_instrument_prompt(rec, pdf_text, results))
            cache[instrument_key] = instrument
        except Exception as e:
            instrument = {"found": False, "preamble": None, "treatment_variations": [],
                          "outcome_questions": [], "source_location": None,
                          "error": str(e)}

    # Post-process: assign arm_ids and cross-match
    if instrument.get("found"):
        instrument = assign_arm_ids(instrument)
        results    = match_result_arm_ids(results, instrument)

    return {
        **rec,
        "extract_status": "ok",
        "pdf_files":      [str(p) for p in pdfs],
        "results":        results,
        "instrument":     instrument,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(model: str, limit: int, only_seq: int | None):
    records = []
    with open(INPUT) as f:
        for line in f:
            r = json.loads(line)
            if r.get("llm_tier") == 1:
                if only_seq is None or r["seq_id"] == only_seq:
                    records.append(r)

    print(f"Tier 1 studies to process: {len(records)}")

    # Partition by PDF availability
    with_pdf    = [r for r in records if find_pdfs_for_study(r["seq_id"])]
    without_pdf = [r for r in records if not find_pdfs_for_study(r["seq_id"])]
    print(f"  With PDF  : {len(with_pdf)}")
    print(f"  No PDF    : {len(without_pdf)}  {[r['seq_id'] for r in without_pdf]}")

    if limit:
        with_pdf = with_pdf[:limit]

    cache  = load_cache()
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(MAX_CONCURRENT)

    tasks   = [extract_one(client, sem, r, cache, model) for r in with_pdf]
    results_list = []
    done = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results_list.append(result)
        done += 1
        save_cache(cache)
        n_out = len((result.get("results") or {}).get("outcomes", []))
        inst  = (result.get("instrument") or {}).get("found", False)
        n_var = len((result.get("instrument") or {}).get("treatment_variations", []))
        print(f"  [{done:>2}/{len(tasks)}] seq={result['seq_id']:>3}  "
              f"outcomes={n_out}  instrument={'yes' if inst else 'no '}  "
              f"arms={n_var}  {result.get('title','')[:50]}")

    # Append no-PDF stubs
    for r in without_pdf:
        results_list.append({**r, "extract_status": "no_pdf", "pdf_files": [],
                              "results": None, "instrument": None})

    results_list.sort(key=lambda r: r["seq_id"])

    # Load existing output to merge (avoid overwriting unprocessed studies)
    existing: dict[int, dict] = {}
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for line in f:
                r = json.loads(line)
                existing[r["seq_id"]] = r

    for r in results_list:
        existing[r["seq_id"]] = r

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for r in sorted(existing.values(), key=lambda x: x["seq_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok_count   = sum(1 for r in results_list if r.get("extract_status") == "ok")
    inst_count = sum(1 for r in results_list if (r.get("instrument") or {}).get("found"))
    print(f"\nExtracted : {ok_count}/{len(results_list)}")
    print(f"Instrument found : {inst_count}/{len(results_list)}")
    print(f"Wrote → {OUTPUT}")


async def run_pass2(model: str, limit: int, only_seq: int | None):
    records = []
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for line in f:
                r = json.loads(line)
                if r.get("extract_status") == "ok" and r.get("results"):
                    if only_seq is None or r["seq_id"] == only_seq:
                        records.append(r)
    
    print(f"Total extracted records to process for pass2: {len(records)}")
    if limit:
        records = records[:limit]

    cache  = load_cache()
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(MAX_CONCURRENT)

    async def call_gpt_pass2(rec: dict) -> dict:
        seq_id = rec["seq_id"]
        pdfs = find_pdfs_for_study(seq_id)
        if not pdfs:
            return rec
            
        pass2_key = f"{seq_id}:pass2_missing_stats"
        print(f"Extracting pass2 stats for seq_id {seq_id}...")
        
        cached = cache.get(pass2_key)
        if cached and not ("rate_limit" in str(cached).lower() or "rate limit" in str(cached).lower() or "429" in str(cached)):
            new_stats = cached
        else:
            pdf_text = extract_pdf_text(pdfs)
            try:
                # Same call structure
                max_retries = 10
                for attempt in range(max_retries):
                    try:
                        async with sem:
                            response = await client.chat.completions.create(
                                model=model,
                                temperature=0,
                                response_format={"type": "json_object"},
                                messages=[
                                    {"role": "system", "content": SECOND_PASS_SYSTEM},
                                    {"role": "user",   "content": make_second_pass_prompt(rec, pdf_text)},
                                ],
                            )
                        new_stats = parse_json_response(response.choices[0].message.content)
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        if "rate_limit" in err_str or "rate limit" in err_str or "429" in err_str:
                            wait_time = min(2 ** attempt + 3, 120)
                            print(f"Rate limit hit for seq_id {seq_id}, attempt {attempt+1}. Waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            raise e
                cache[pass2_key] = new_stats
            except Exception as e:
                print(f"Pass2 failed for {seq_id}: {e}")
                return rec
                
        # Merge new_stats back into rec
        if new_stats and new_stats.get("outcomes"):
            # Build lookups
            stats_lookup = {}
            for out in new_stats["outcomes"]:
                oname = out.get("name")
                for gs in out.get("group_summaries", []):
                    stats_lookup[(oname, gs.get("arm_label"))] = gs
                    
            # Apply to rec
            for out in rec["results"].get("outcomes", []):
                oname = out.get("name")
                for gs in out.get("group_summaries", []):
                    key = (oname, gs.get("arm_label"))
                    if key in stats_lookup:
                        new_gs = stats_lookup[key]
                        if new_gs.get("n_analyzed") is not None: gs["n_analyzed"] = new_gs["n_analyzed"]
                        if new_gs.get("sd") is not None: gs["sd"] = new_gs["sd"]
                        if new_gs.get("se") is not None: gs["se"] = new_gs["se"]
                        
        return rec

    tasks = [call_gpt_pass2(r) for r in records]
    
    updated_records = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        res = await coro
        updated_records.append(res)
        done += 1
        save_cache(cache)
        print(f"  [{done:>2}/{len(tasks)}] pass2 seq={res['seq_id']:>3}")

    # Load any existing ones we skipped so we just duplicate the original file but inject updates
    final_records = []
    updated_seqs = {r["seq_id"]: r for r in updated_records}
    
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for line in f:
                r = json.loads(line)
                seq_id = r["seq_id"]
                if seq_id in updated_seqs:
                    final_records.append(updated_seqs[seq_id])
                else:
                    final_records.append(r)

    OUTPUT_PASS2.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PASS2, "w") as f:
        for r in sorted(final_records, key=lambda x: x["seq_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\nWrote (pass2) → {OUTPUT_PASS2}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process first N Tier 1 PDFs only (0 = all)")
    parser.add_argument("--seq",   type=int, default=None,
                        help="Process only this seq_id (for debugging)")
    parser.add_argument("--pass2", action="store_true",
                        help="Run second pass over extracted data to find missing n and sd")
    args = parser.parse_args()
    
    if args.pass2:
        asyncio.run(run_pass2(args.model, args.limit, args.seq))
    else:
        asyncio.run(run(args.model, args.limit, args.seq))


if __name__ == "__main__":
    main()
