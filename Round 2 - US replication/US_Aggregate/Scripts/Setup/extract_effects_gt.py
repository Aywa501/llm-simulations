"""
extract_effects_gt.py

For each study in simulatable_studies.jsonl that has a control_arm_id,
reads the paper PDF and asks GPT to extract treatment effects:

    Δ = mean(treatment_arm) − mean(control_arm)

for every (non-control arm) × (outcome) combination.

Critically, the extraction uses the *exact* arm_ids and outcome_names from
the instrument — so the output shares the same identifiers as the simulation,
removing the need for any fuzzy matching at comparison time.

Output: Data/Ground_Truth/study_effects_gt.jsonl
  One record per study with fields:
    seq_id, title, control_arm_id, extract_status, effects[]

Each effect entry:
    arm_id          — from instrument (exact)
    outcome_id      — slugified outcome_name (matches simulate_aggregate.py)
    outcome_name    — human-readable
    delta           — treatment_mean − control_mean  (primary comparison value)
    treatment_mean  — raw mean for treatment arm (may be null)
    control_mean    — raw mean for control arm   (may be null)
    n_treatment     — sample size in treatment arm
    n_control       — sample size in control arm
    metric          — mean | proportion | coefficient | other
    note            — extraction caveats

Usage:
    python extract_effects_gt.py [--model gpt-4.1] [--force]
"""

import argparse, asyncio, json, re, time
from pathlib import Path
from typing import Any

import pdfplumber
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR.parents[1] / "Data"
STUDIES_PATH = DATA_DIR / "Ground_Truth" / "simulatable_studies.jsonl"
PAPERS_ROOT  = DATA_DIR / "Papers"
OUTPUT_PATH  = DATA_DIR / "Ground_Truth" / "study_effects_gt.jsonl"
CACHE_PATH   = DATA_DIR / "Caches" / ".extract_effects_cache.json"

MAX_PDF_CHARS  = 90_000
MAX_CONCURRENT = 4
DEFAULT_MODEL  = "gpt-4.1"

# Thresholds for coverage grading
COVERAGE_OK      = 0.75   # ≥75 % of expected (arm, outcome) pairs extracted with non-null delta
COVERAGE_PARTIAL = 0.25   # ≥25 %

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default=DEFAULT_MODEL)
parser.add_argument("--force",  action="store_true",
                    help="Re-extract even if result is already cached")
parser.add_argument("--seq-ids", nargs="*", type=int,
                    help="Only process these seq_ids (default: all)")
args = parser.parse_args()

client = AsyncOpenAI()

# ---------------------------------------------------------------------------
# Slugify (must match simulate_aggregate.py exactly)
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50]

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def find_pdfs(seq_id: int) -> list[Path]:
    """Return primary PDFs for a study (excludes *_supplement.pdf)."""
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

# ---------------------------------------------------------------------------
# Load studies
# ---------------------------------------------------------------------------

def load_studies() -> list[dict]:
    studies = []
    with open(STUDIES_PATH) as f:
        for line in f:
            rec = json.loads(line)
            inst = rec.get("instrument", {})
            # Skip studies with no control arm defined
            if inst.get("control_arm_id") is None:
                continue
            studies.append(rec)
    return studies

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise research assistant that extracts quantitative results "
    "from social science papers. You return only valid JSON with no prose."
)


def build_prompt(study: dict, paper_text: str) -> str:
    inst         = study.get("instrument", {})
    control_arm  = inst.get("control_arm_id", "")
    arms         = inst.get("treatment_variations", [])
    outcomes     = inst.get("outcome_questions", [])
    title        = study.get("title", "")

    # List non-control arms
    treatment_arms = [a for a in arms if a.get("arm_id") != control_arm]

    arm_lines = "\n".join(
        f'  arm_id="{a["arm_id"]}"  label="{a.get("arm_label", "")}"'
        for a in treatment_arms
    )
    ctrl_arm_obj = next((a for a in arms if a.get("arm_id") == control_arm), {})
    ctrl_label   = ctrl_arm_obj.get("arm_label", control_arm)

    outcome_lines = "\n".join(
        f'  outcome_id="{slugify(o.get("outcome_name", ""))}"  '
        f'name="{o.get("outcome_name", "")}"'
        for o in outcomes
    )

    expected_n = len(treatment_arms) * len(outcomes)

    return f"""You are extracting experimental results from the following paper.

STUDY TITLE: {title}

CONTROL ARM (baseline):
  arm_id="{control_arm}"  label="{ctrl_label}"

TREATMENT ARMS (each compared against the control):
{arm_lines}

OUTCOME MEASURES:
{outcome_lines}

TASK
For each combination of (treatment arm × outcome measure), locate the
relevant result in the paper and extract:

  treatment_mean  — mean / proportion / score for the treatment arm
  control_mean    — mean / proportion / score for the control arm
  delta           — treatment_mean MINUS control_mean
  n_treatment     — number of participants in the treatment arm
  n_control       — number of participants in the control arm
  metric          — one of: "mean", "proportion", "coefficient", "other"
  note            — brief extraction caveat (e.g. "from Table 3 col 2",
                    "regression coefficient not raw mean", "imputed from
                    graph") or empty string

Rules:
• Use the arm_id and outcome_id strings EXACTLY as given above.
• If the paper reports a regression coefficient for the treatment arm
  instead of raw means, set delta = coefficient, treatment_mean = null,
  control_mean = null, metric = "coefficient".
• If a value genuinely cannot be found, set it to null — do NOT guess.
• You must return exactly {expected_n} effect objects (one per arm × outcome).

Return ONLY a JSON object with this structure — no markdown, no prose:
{{
  "effects": [
    {{
      "arm_id": "<arm_id>",
      "outcome_id": "<outcome_id>",
      "treatment_mean": <number or null>,
      "control_mean": <number or null>,
      "delta": <number or null>,
      "n_treatment": <integer or null>,
      "n_control": <integer or null>,
      "metric": "<mean|proportion|coefficient|other>",
      "note": "<string>"
    }}
  ]
}}

PAPER TEXT:
{paper_text}"""


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

async def call_with_retry(prompt: str, model: str, retries: int = 6) -> str | None:
    import random
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content
        except RateLimitError:
            wait = min(20 * (2 ** attempt), 120) + random.uniform(0, 5)
            print(f"    Rate limit — waiting {wait:.1f}s (attempt {attempt+1})")
            await asyncio.sleep(wait)
        except APIConnectionError:
            wait = min(5 * (2 ** attempt), 30) + random.uniform(0, 3)
            print(f"    Connection error — waiting {wait:.1f}s")
            await asyncio.sleep(wait)
        except APIError as e:
            wait = min(10 * (2 ** attempt), 60) + random.uniform(0, 5)
            print(f"    API error ({e}) — waiting {wait:.1f}s")
            await asyncio.sleep(wait)
    return None

# ---------------------------------------------------------------------------
# Parse LLM response
# ---------------------------------------------------------------------------

def parse_effects_json(raw: str | None) -> list[dict] | None:
    """Extract the effects list from the LLM's JSON response."""
    if not raw:
        return None
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        obj = json.loads(raw)
        effects = obj.get("effects", [])
        if not isinstance(effects, list):
            return None
        return effects
    except json.JSONDecodeError:
        # Try to find a JSON object in the response
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            try:
                return json.loads(m.group())["effects"]
            except Exception:
                pass
        return None

# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def coverage(effects: list[dict], expected_n: int) -> float:
    if expected_n == 0:
        return 1.0
    found = sum(1 for e in effects if e.get("delta") is not None)
    return found / expected_n

# ---------------------------------------------------------------------------
# Cache helpers
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
# Per-study extraction
# ---------------------------------------------------------------------------

async def extract_study(study: dict, cache: dict,
                        sem: asyncio.Semaphore) -> dict:
    seq_id = study["seq_id"]
    title  = study.get("title", f"seq={seq_id}")
    inst   = study.get("instrument", {})
    ctrl   = inst.get("control_arm_id")
    arms   = [a for a in inst.get("treatment_variations", [])
              if a.get("arm_id") != ctrl]
    outcomes = inst.get("outcome_questions", [])
    expected_n = len(arms) * len(outcomes)

    cache_key = f"effects__{seq_id}__{args.model}"

    # Return cached result if available and not forced
    if not args.force and cache_key in cache:
        raw_effects = cache[cache_key]
        status = _grade_status(raw_effects, expected_n)
        print(f"  seq={seq_id:>3}  [CACHED]  {status}  "
              f"({sum(1 for e in raw_effects if e.get('delta') is not None)}"
              f"/{expected_n} deltas)")
        return _build_record(study, raw_effects, status)

    # Find and read PDFs
    pdf_paths = find_pdfs(seq_id)
    if not pdf_paths:
        print(f"  seq={seq_id:>3}  [NO PDF]  skipping")
        return _build_record(study, [], "failed")

    paper_text = extract_pdf_text(pdf_paths)
    prompt     = build_prompt(study, paper_text)

    async with sem:
        print(f"  seq={seq_id:>3}  extracting ({len(arms)} arms × "
              f"{len(outcomes)} outcomes = {expected_n} expected)…")
        raw_text = await call_with_retry(prompt, args.model)

    effects = parse_effects_json(raw_text) or []

    # Retry if coverage is too low
    if coverage(effects, expected_n) < COVERAGE_PARTIAL and expected_n > 0:
        print(f"  seq={seq_id:>3}  low coverage — retrying…")
        async with sem:
            raw_text2 = await call_with_retry(prompt, args.model)
        effects2 = parse_effects_json(raw_text2) or []
        if coverage(effects2, expected_n) > coverage(effects, expected_n):
            effects = effects2

    # Enrich: add outcome_name to each effect from the instrument
    slug_to_name = {
        slugify(o.get("outcome_name", "")): o.get("outcome_name", "")
        for o in outcomes
    }
    for e in effects:
        e["outcome_name"] = slug_to_name.get(e.get("outcome_id", ""), "")

    status = _grade_status(effects, expected_n)
    n_found = sum(1 for e in effects if e.get("delta") is not None)
    print(f"  seq={seq_id:>3}  {status}  ({n_found}/{expected_n} deltas)")

    cache[cache_key] = effects
    save_cache(cache)

    return _build_record(study, effects, status)


def _grade_status(effects: list[dict], expected_n: int) -> str:
    cov = coverage(effects, expected_n)
    if cov >= COVERAGE_OK:
        return "ok"
    elif cov >= COVERAGE_PARTIAL:
        return "partial"
    else:
        return "failed"


def _build_record(study: dict, effects: list[dict], status: str) -> dict:
    inst = study.get("instrument", {})
    return {
        "seq_id":          study["seq_id"],
        "title":           study.get("title", ""),
        "control_arm_id":  inst.get("control_arm_id"),
        "extract_status":  status,
        "effects":         effects,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    studies = load_studies()

    if args.seq_ids:
        studies = [s for s in studies if s["seq_id"] in args.seq_ids]

    if not studies:
        print("No studies to process (check control_arm_id fields — "
              "run add_control_arms.py first if needed).")
        return

    print(f"Processing {len(studies)} studies with control arms…")
    print(f"Model: {args.model}  |  force={args.force}\n")

    cache = load_cache()
    sem   = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [extract_study(s, cache, sem) for s in studies]
    results = await asyncio.gather(*tasks)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    ok      = sum(1 for r in results if r["extract_status"] == "ok")
    partial = sum(1 for r in results if r["extract_status"] == "partial")
    failed  = sum(1 for r in results if r["extract_status"] == "failed")
    total_effects = sum(
        sum(1 for e in r["effects"] if e.get("delta") is not None)
        for r in results
    )

    print(f"\n{'='*55}")
    print(f"Done.  ok={ok}  partial={partial}  failed={failed}")
    print(f"Total non-null deltas extracted: {total_effects}")
    print(f"Output → {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
