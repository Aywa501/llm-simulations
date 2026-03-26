"""
extract_from_paper.py

Step 4 of the aggregate pipeline.

For each Tier 1 study with a downloaded PDF, extracts two things via GPT:
  (A) Aggregate results  — per-arm N, outcome name, metric, value, SD/SE
  (B) Instrument text    — the actual survey question / experimental stimulus
                           shown to participants

PDF text is extracted with pdfplumber before being sent to the model.

Outputs Data/study_enriched_aggregate.jsonl, one record per study.

Requires:
  - OPENAI_API_KEY in environment
  - pip install pdfplumber
  - Data/tier_classified.jsonl  (from classify_tiers_llm.py)
  - Data/papers/{seq_id}.pdf    (from fetch_papers.py)

Run from this script's directory:
    python extract_from_paper.py [--limit N] [--model MODEL]
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

import pdfplumber
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).resolve().parents[1] / "Data"
INPUT      = DATA_DIR / "tier_classified.jsonl"
PAPERS_DIR = DATA_DIR / "papers"
OUTPUT     = DATA_DIR / "study_enriched_aggregate.jsonl"
CACHE_FILE = DATA_DIR / ".extract_cache.json"

DEFAULT_MODEL  = "gpt-5.4-mini"
MAX_CONCURRENT = 5   # GPT-4o with large contexts — keep concurrency moderate
MAX_PDF_CHARS  = 60_000  # truncate extracted text to stay within context limits

# ---------------------------------------------------------------------------
# JSON schemas (sent as prompt instructions, not enforced by API)
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
          "arm_label": "<arm name as in paper>",
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
  "extraction_notes": "<caveats, ambiguities, or assumptions>"
}"""

INSTRUMENT_SCHEMA_DESC = """
{
  "found": true | false,
  "instrument_text": "<verbatim or near-verbatim text shown to all participants, or null>",
  "treatment_variations": [
    {"arm_label": "<arm name>", "text": "<arm-specific stimulus text>"}
  ],
  "source_location": "<e.g. Appendix A, p.8 — or null>"
}"""

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RESULTS_SYSTEM = (
    "You are extracting experimental results from a social science paper. "
    "Extract the main results for the PRIMARY outcome(s) only. "
    "Focus on the simplest direct comparison between treatment and control arms — "
    "typically the first main results table. "
    "Report numbers exactly as they appear. Use null for any value not reported. "
    "Respond with valid JSON only — no markdown fences, no explanation outside JSON."
)

INSTRUMENT_SYSTEM = (
    "You are extracting the experimental stimulus or survey instrument from a social "
    "science paper. Look in the main text and appendices. "
    "The instrument is the verbatim text shown to participants: survey questions, "
    "vignettes, informational treatments, or experimental instructions. "
    "If different arms received different text, list each variant. "
    "If no instrument text appears in the paper, set found=false. "
    "Respond with valid JSON only — no markdown fences, no explanation outside JSON."
)


def make_results_prompt(rec: dict, pdf_text: str) -> str:
    return (
        f"STUDY: {rec.get('title', '')}\n\n"
        f"INTERVENTION: {rec.get('intervention_text', '')[:600]}\n\n"
        f"EXPERIMENTAL DESIGN: {rec.get('experimental_design', '')[:600]}\n\n"
        f"PRIMARY OUTCOMES: {rec.get('primary_outcomes', '')[:300]}\n\n"
        f"--- PAPER TEXT ---\n{pdf_text}\n--- END PAPER ---\n\n"
        f"Extract the main results following this JSON schema:\n{RESULTS_SCHEMA_DESC}"
    )


def make_instrument_prompt(rec: dict, pdf_text: str) -> str:
    return (
        f"STUDY: {rec.get('title', '')}\n\n"
        f"INTERVENTION: {rec.get('intervention_text', '')[:600]}\n\n"
        f"EXPERIMENTAL DESIGN: {rec.get('experimental_design', '')[:600]}\n\n"
        f"--- PAPER TEXT ---\n{pdf_text}\n--- END PAPER ---\n\n"
        f"Extract the experimental instrument following this JSON schema:\n{INSTRUMENT_SCHEMA_DESC}"
    )


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(path: Path) -> str:
    """Extract all text from a PDF using pdfplumber, up to MAX_PDF_CHARS."""
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
    except Exception as e:
        return f"[PDF extraction failed: {e}]"

    full = "\n\n".join(pages)
    if len(full) > MAX_PDF_CHARS:
        full = full[:MAX_PDF_CHARS] + "\n[... truncated ...]"
    return full


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_key(seq_id: int, extraction_type: str) -> str:
    return f"{seq_id}:{extraction_type}"


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

async def extract_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    rec: dict,
    cache: dict,
    model: str,
) -> dict:
    seq_id   = rec["seq_id"]
    pdf_path = PAPERS_DIR / f"{seq_id}.pdf"

    if not pdf_path.exists():
        return {**rec, "extract_status": "no_pdf", "results": None, "instrument": None}

    pdf_text = extract_pdf_text(pdf_path)

    results_key    = cache_key(seq_id, "results")
    instrument_key = cache_key(seq_id, "instrument")

    async def call_gpt(system: str, user: str) -> dict:
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
        return parse_json(response.choices[0].message.content)

    # Results
    if results_key in cache:
        results = cache[results_key]
    else:
        try:
            results = await call_gpt(RESULTS_SYSTEM, make_results_prompt(rec, pdf_text))
            cache[results_key] = results
        except Exception as e:
            results = {"outcomes": [], "extraction_notes": f"failed: {e}"}

    # Instrument
    if instrument_key in cache:
        instrument = cache[instrument_key]
    else:
        try:
            instrument = await call_gpt(INSTRUMENT_SYSTEM, make_instrument_prompt(rec, pdf_text))
            cache[instrument_key] = instrument
        except Exception as e:
            instrument = {"found": False, "instrument_text": None,
                          "treatment_variations": [], "source_location": None,
                          "error": str(e)}

    return {
        **rec,
        "extract_status": "ok",
        "results":        results,
        "instrument":     instrument,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(model: str, limit: int):
    records = []
    with open(INPUT) as f:
        for line in f:
            r = json.loads(line)
            if r.get("llm_tier") == 1:
                records.append(r)

    print(f"Tier 1 studies: {len(records)}")

    with_pdf    = [r for r in records if (PAPERS_DIR / f"{r['seq_id']}.pdf").exists()]
    without_pdf = [r for r in records if r not in with_pdf]
    print(f"  With downloaded PDF : {len(with_pdf)}")
    print(f"  Missing PDF (skip)  : {len(without_pdf)}")

    if limit:
        with_pdf = with_pdf[:limit]

    cache  = load_cache()
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(MAX_CONCURRENT)

    tasks   = [extract_one(client, sem, r, cache, model) for r in with_pdf]
    results = []
    done    = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        done += 1
        save_cache(cache)
        n_out  = len((result.get("results") or {}).get("outcomes", []))
        inst   = (result.get("instrument") or {}).get("found", False)
        print(f"  [{done}/{len(tasks)}] seq={result['seq_id']:>3}  "
              f"outcomes={n_out}  instrument={'yes' if inst else 'no '}  "
              f"{result.get('title','')[:55]}")

    # Append no-PDF stubs
    for r in without_pdf:
        results.append({**r, "extract_status": "no_pdf", "results": None, "instrument": None})

    results.sort(key=lambda r: r["seq_id"])

    with open(OUTPUT, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok_count   = sum(1 for r in results if r.get("extract_status") == "ok")
    inst_count = sum(1 for r in results if (r.get("instrument") or {}).get("found"))
    print(f"\nResults extracted : {ok_count}/{len(results)}")
    print(f"Instrument found  : {inst_count}/{len(results)}")
    print(f"Wrote → {OUTPUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process first N Tier 1 PDFs only (0=all)")
    args = parser.parse_args()
    asyncio.run(run(args.model, args.limit))


if __name__ == "__main__":
    main()
