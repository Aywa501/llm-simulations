"""
extract_yellow_instruments.py

Extracts instrument text and confirms results from the 4 yellow study PDFs.
Reads study context from the Track 1 JSONL, extracts from PDFs in llm-simulations-local/Papers/,
and writes to Data/yellow_instruments.jsonl.

Run from this script's directory:
    python extract_yellow_instruments.py
"""

import asyncio
import json
import os
import re
from pathlib import Path

import pdfplumber
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_DIR  = Path(__file__).resolve().parent                              # Scripts/Microdata/
DATA_DIR     = SCRIPTS_DIR.parents[1] / "Data" / "Microdata"
PAPERS_DIR   = SCRIPTS_DIR.parents[3] / "llm-simulations-local" / "Papers"
TRACK1_JSONL = DATA_DIR / "SORTED DATA - study_enriched_tier1and2_tagged.jsonl"
OUTPUT       = DATA_DIR / "yellow_instruments.jsonl"

MODEL        = "gpt-5.4-mini"
MAX_PDF_CHARS = 80_000

# ---------------------------------------------------------------------------
# Study → PDF mapping  (seq_id from Track 1 → pdf filename)
# ---------------------------------------------------------------------------
YELLOW_STUDIES = {
    44: {
        "rct_id": "AEARCTR-0004316",
        "pdf": "ssrn-3450207.pdf",
        "label": "Complexity and Expectation Formation",
    },
    45: {
        "rct_id": "AEARCTR-0004651",
        "pdf": "2407.14955v4.pdf",
        "label": "Temptation: Immediacy and certainty",
    },
    46: {
        "rct_id": "AEARCTR-0005064",
        "pdf": "fpsyg-12-675776.pdf",
        "label": "Attitudes towards hiring decisions",
    },
    57: {
        "rct_id": "AEARCTR-0012106",
        "pdf": "Behavioral Food Subsidies _{Brownback, Andy (author)_Imas, Alex (author)_Kuhn, Michael A (author)}(2019){108805250} libgen.li.pdf",
        "label": "Time Preferences and Food Choice",
    },
}

# ---------------------------------------------------------------------------
# Load Track 1 study context for each yellow study
# ---------------------------------------------------------------------------

def load_track1_records() -> dict:
    records = {}
    with open(TRACK1_JSONL) as f:
        for line in f:
            r = json.loads(line)
            seq = r.get("seq_id")
            if seq in YELLOW_STUDIES:
                records[seq] = r
    return records


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(path: Path) -> str:
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
# Prompts
# ---------------------------------------------------------------------------

INSTRUMENT_SYSTEM = (
    "You are extracting the experimental stimulus or survey instrument from a social "
    "science paper. Look in the main text and appendices. "
    "The instrument is the verbatim (or near-verbatim) text shown to participants: "
    "survey questions, vignettes, informational treatments, choice scenarios, or "
    "experimental task instructions. "
    "If different arms received different text, list each variant separately. "
    "If no instrument text appears in the paper, set found=false. "
    "Respond with valid JSON only — no markdown fences."
)

INSTRUMENT_SCHEMA = """{
  "found": true | false,
  "instrument_text": "<shared text shown to all participants, or null>",
  "treatment_variations": [
    {"arm_label": "<arm name>", "text": "<arm-specific text shown to participants>"}
  ],
  "response_format": "<how participants responded: e.g. 'binary choice', 'likert 1-7', 'dollar amount 0-3', 'numeric forecast'>",
  "source_location": "<e.g. Appendix A, p.8>"
}"""

RESULTS_SYSTEM = (
    "You are extracting the main experimental results from a social science paper. "
    "Find the primary outcome table and extract per-arm summary statistics. "
    "Respond with valid JSON only — no markdown fences."
)

RESULTS_SCHEMA = """{
  "outcomes": [
    {
      "name": "<outcome name>",
      "is_primary": true | false,
      "outcome_type": "binary" | "continuous" | "count" | "ordinal" | "other",
      "table_reference": "<e.g. Table 2>",
      "group_summaries": [
        {
          "arm_label": "<arm name as in paper>",
          "n_analyzed": <int or null>,
          "metric": "mean" | "proportion" | "rate" | "other",
          "value": <number or null>,
          "sd": <number or null>,
          "se": <number or null>
        }
      ]
    }
  ],
  "extraction_notes": "<any caveats>"
}"""


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    return json.loads(text)


def make_context(rec: dict) -> str:
    reg = rec.get("provenance", {}).get("registry", {})
    return (
        f"TITLE: {reg.get('title', '')}\n"
        f"INTERVENTION: {str(reg.get('intervention_text', ''))[:600]}\n"
        f"EXPERIMENTAL DESIGN: {str(reg.get('experimental_design', ''))[:600]}\n"
        f"PRIMARY OUTCOMES: {str(reg.get('primary_outcomes', ''))[:300]}"
    )


async def extract_one(client: AsyncOpenAI, seq_id: int, meta: dict, track1_rec: dict) -> dict:
    pdf_path = PAPERS_DIR / meta["pdf"]
    if not pdf_path.exists():
        print(f"  [seq={seq_id}] PDF not found: {pdf_path}")
        return {"seq_id": seq_id, **meta, "instrument": None, "results": None, "status": "pdf_missing"}

    print(f"  [seq={seq_id}] Extracting from {meta['pdf'][:50]}...")
    pdf_text = extract_pdf_text(pdf_path)
    context  = make_context(track1_rec) if track1_rec else f"TITLE: {meta['label']}"

    async def call(system, schema_desc):
        resp = await client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    f"{context}\n\n"
                    f"--- PAPER TEXT ---\n{pdf_text}\n--- END ---\n\n"
                    f"Extract following this schema:\n{schema_desc}"
                )},
            ],
        )
        return parse_json(resp.choices[0].message.content)

    try:
        instrument = await call(INSTRUMENT_SYSTEM, INSTRUMENT_SCHEMA)
    except Exception as e:
        instrument = {"found": False, "error": str(e)}

    try:
        results = await call(RESULTS_SYSTEM, RESULTS_SCHEMA)
    except Exception as e:
        results = {"outcomes": [], "extraction_notes": str(e)}

    return {
        "seq_id":     seq_id,
        "rct_id":     meta["rct_id"],
        "label":      meta["label"],
        "status":     "ok",
        "instrument": instrument,
        "results":    results,
    }


async def main():
    track1 = load_track1_records()
    print(f"Track 1 records loaded: {list(track1.keys())}")

    client = AsyncOpenAI()
    tasks  = [
        extract_one(client, seq_id, meta, track1.get(seq_id, {}))
        for seq_id, meta in YELLOW_STUDIES.items()
    ]
    outputs = await asyncio.gather(*tasks)

    with open(OUTPUT, "w") as f:
        for r in outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Results → {OUTPUT}")
    for r in outputs:
        inst = r.get("instrument") or {}
        print(f"  seq={r['seq_id']} {r['label'][:50]}")
        print(f"    instrument found={inst.get('found')}  "
              f"response_format={inst.get('response_format','?')}  "
              f"variants={len(inst.get('treatment_variations', []))}")


if __name__ == "__main__":
    asyncio.run(main())
