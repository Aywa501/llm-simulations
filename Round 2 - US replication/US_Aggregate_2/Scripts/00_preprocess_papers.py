"""
00_preprocess_papers.py  —  Step 0 of the pipeline (optional but recommended)

Pre-processes raw PDF text using gpt-5.4-mini before the main extraction step.

WHY
---
Study PDFs can be 60–90k characters of raw pdfplumber output.  Most of that
text is irrelevant: author affiliations, literature reviews, discussion,
references, funding statements.  The main extraction model (gpt-5.4) wastes
context window and attention on all of it.

This script sends the raw PDF text through a cheap fast model that:
  1. Keeps verbatim any stimulus text / vignettes / messages shown to participants
  2. Keeps verbatim all outcome measure questions and scale descriptions
  3. Keeps the methods/design section and all results tables/statistics
  4. Summarises lengthy literature review paragraphs (1–2 sentences each)
  5. Drops author affiliations, acknowledgements, funding, full reference lists,
     and any appendices that don't contain participant-facing materials

The resulting condensed text is typically 30–60% of the original length while
preserving everything the extraction model needs.  It is saved to a cache file
and automatically picked up by 01_extract_study_data.py.

Output
------
  Data/Caches/.preprocessed_papers.json
  {seq_id: {"text": "<condensed text>", "original_chars": N, "condensed_chars": M}}

Usage
-----
  # Process all studies found in Data/Papers/
  python 00_preprocess_papers.py

  # Process specific studies
  python 00_preprocess_papers.py --seq-ids 12 19 30

  # Force re-process (bypass cache)
  python 00_preprocess_papers.py --force

  # Change model (default: gpt-5.4-mini)
  python 00_preprocess_papers.py --model gpt-4.1-mini
"""

import argparse, asyncio, json, re
from pathlib import Path

import pdfplumber
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR.parent / "Data"
PAPERS_ROOT  = DATA_DIR / "Papers"
CACHE_PATH   = DATA_DIR / "Caches" / ".preprocessed_papers.json"

MAX_PDF_CHARS  = 120_000   # generous raw input limit
MAX_CONCURRENT = 3         # conservative — avoids rate-limit cascade on large papers
DEFAULT_MODEL  = "gpt-5.4-mini"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model",    default=DEFAULT_MODEL,
                    help=f"Model to use for preprocessing (default: {DEFAULT_MODEL})")
parser.add_argument("--force",    action="store_true",
                    help="Re-process even if already cached")
parser.add_argument("--seq-ids",  nargs="*", type=int,
                    help="Only process these seq_ids (default: all PDFs found)")
args = parser.parse_args()

client = AsyncOpenAI()

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a precise research assistant preprocessing a social science paper "
    "for downstream data extraction. Return only the condensed text — "
    "no commentary, no JSON, no markdown fencing."
)

PREPROCESS_PROMPT = """\
Extract a condensed version of this social science paper containing ONLY the \
information needed to:
  (A) identify experimental conditions and which is the control
  (B) recover exact stimulus / vignette / message text shown to participants
  (C) identify outcome measures: exact question wording and response scale
  (D) recover reported statistics: means, SDs, proportions, n, p-values, CIs, \
effect sizes

TARGET: roughly 25–35% of the original length.  Be aggressive.

━━━ KEEP VERBATIM (copy exactly, do not paraphrase) ━━━
• Every word of stimulus / vignette / message / scenario text shown to participants
• Every survey question and its response scale / anchors
• The Methods section: design, conditions, procedure, participant details
• All results: tables, in-text statistics (M=, SD=, n=, p=, CI, β, d, F, t, χ²)
• Abstract (keep as-is; it is short)

━━━ COMPRESS to 1–2 sentences ━━━
• Each literature review paragraph — one sentence capturing only the claim \
that directly motivates the study design
• Discussion / implications — two sentences max on main takeaway
• Any repeated descriptions once summarised the first time

━━━ DROP ENTIRELY (output nothing for these) ━━━
• Author names, affiliations, emails, ORCIDs, institutional addresses
• Acknowledgements, funding disclosures, conflict of interest statements
• Full reference list (in-text citations like "(Smith, 2020)" are fine to keep)
• Footnotes and endnotes that do not contain stimulus text or statistics
• Page headers, footers, journal name, volume, DOI metadata
• Appendices containing only supplementary regression tables or robustness \
checks (unless the main results reference them as primary)
• Any sentence that is purely methodological boilerplate with no study-specific \
content (e.g. "Participants gave informed consent prior to the study.")

━━━ OUTPUT FORMAT ━━━
Plain text.  Preserve section headings.  No commentary, no JSON, no markdown.
The downstream model extracting data from your output will need every kept word \
to be accurate — do not alter any numbers, question text, or condition labels.

PAPER TEXT:
{paper_text}"""

# ---------------------------------------------------------------------------
# PDF helpers  (identical logic to 01_extract_study_data.py)
# ---------------------------------------------------------------------------

def find_pdfs(seq_id: int) -> list[Path]:
    found = []
    for f in PAPERS_ROOT.rglob("*.pdf"):
        stem = f.stem
        if stem == str(seq_id):
            found.append(f)
        elif re.match(rf"^{seq_id}_\d+$", stem):
            found.append(f)
    return sorted(found)


def extract_pdf_text(paths: list[Path]) -> str:
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


def discover_all_seq_ids() -> list[int]:
    seen = set()
    for f in PAPERS_ROOT.rglob("*.pdf"):
        m = re.match(r"^(\d+)", f.stem)
        if m:
            seen.add(int(m.group(1)))
    return sorted(seen)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

async def call_api(prompt: str, retries: int = 8) -> str | None:
    import random
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=args.model,
                temperature=0,
                max_completion_tokens=16000,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content
        except RateLimitError:
            wait = min(30 * (2 ** attempt), 180) + random.uniform(0, 10)
            print(f"\n    Rate limit — waiting {wait:.0f}s (attempt {attempt+1}/{retries})")
            await asyncio.sleep(wait)
        except APIConnectionError:
            wait = min(10 * (2 ** attempt), 60) + random.uniform(0, 5)
            print(f"\n    Connection error — waiting {wait:.0f}s")
            await asyncio.sleep(wait)
        except APIError as e:
            wait = min(15 * (2 ** attempt), 90) + random.uniform(0, 5)
            print(f"\n    API error ({e}) — waiting {wait:.0f}s")
            await asyncio.sleep(wait)
    return None

# ---------------------------------------------------------------------------
# Per-study preprocessing
# ---------------------------------------------------------------------------

async def preprocess_one(seq_id: int, cache: dict,
                         sem: asyncio.Semaphore) -> dict | None:
    cache_key = f"{seq_id}__{args.model}"

    if not args.force and cache_key in cache:
        entry = cache[cache_key]
        ratio = entry["condensed_chars"] / max(entry["original_chars"], 1)
        print(f"  seq={seq_id:>3}  [CACHED]  "
              f"{entry['original_chars']:>6} → {entry['condensed_chars']:>6} chars  "
              f"({ratio:.0%})")
        return entry

    pdf_paths = find_pdfs(seq_id)
    if not pdf_paths:
        print(f"  seq={seq_id:>3}  [NO PDF]")
        return None

    raw_text = extract_pdf_text(pdf_paths)
    orig_len = len(raw_text)

    async with sem:
        print(f"  seq={seq_id:>3}  preprocessing ({orig_len:,} chars)…",
              end="", flush=True)
        condensed = await call_api(
            PREPROCESS_PROMPT.format(paper_text=raw_text)
        )

    if not condensed:
        print(f"\n  seq={seq_id:>3}  FAILED — keeping raw text as fallback")
        # Store raw text so 01 can still run; flag it
        entry = {
            "seq_id":          seq_id,
            "model":           args.model,
            "text":            raw_text,
            "original_chars":  orig_len,
            "condensed_chars": orig_len,
            "status":          "failed_kept_raw",
        }
    else:
        cond_len = len(condensed)
        ratio    = cond_len / max(orig_len, 1)
        print(f"  {orig_len:>6} → {cond_len:>6} chars  ({ratio:.0%})")
        entry = {
            "seq_id":          seq_id,
            "model":           args.model,
            "text":            condensed,
            "original_chars":  orig_len,
            "condensed_chars": cond_len,
            "status":          "ok",
        }

    cache[cache_key] = entry
    save_cache(cache)
    return entry

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    seq_ids = args.seq_ids or discover_all_seq_ids()
    print(f"Preprocessing {len(seq_ids)} studies  |  "
          f"model={args.model}  force={args.force}\n")

    cache   = load_cache()
    sem     = asyncio.Semaphore(MAX_CONCURRENT)
    tasks   = [preprocess_one(sid, cache, sem) for sid in sorted(seq_ids)]
    results = await asyncio.gather(*tasks)
    results = [r for r in results if r is not None]

    ok      = sum(1 for r in results if r.get("status") == "ok")
    cached  = sum(1 for r in results if r.get("status") not in ("ok", "failed_kept_raw"))
    failed  = sum(1 for r in results if r.get("status") == "failed_kept_raw")

    total_orig  = sum(r["original_chars"]  for r in results)
    total_cond  = sum(r["condensed_chars"] for r in results)
    overall_ratio = total_cond / max(total_orig, 1)

    print(f"\n{'='*60}")
    print(f"Studies processed  : {len(results)}")
    print(f"  Fresh / ok       : {ok}")
    print(f"  Cached           : {cached}")
    print(f"  Failed (raw kept): {failed}")
    print(f"Total original     : {total_orig:,} chars")
    print(f"Total condensed    : {total_cond:,} chars  ({overall_ratio:.0%} of original)")
    print(f"Cache → {CACHE_PATH}")
    print(f"\n01_extract_study_data.py will automatically use this preprocessed text.")


if __name__ == "__main__":
    asyncio.run(main())
