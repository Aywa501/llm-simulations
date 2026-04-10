"""
extract_study_data.py

Top-down extraction pipeline.  For each study PDF, produces a unified record
containing both the simulation instrument and ground-truth treatment effects.

Two passes per study
--------------------
Pass 1 — DESIGN
  LLM reads the paper and returns the experimental design:
    • all conditions (arm_label, arm_id, exact stimulus text)
    • which condition is the control / baseline
    • outcome measures (question text, scale, response format)
    • whether the study is LLM-simulatable
    • a shared preamble shown to all participants

Pass 2 — RESULTS
  Using the exact arm_ids and outcome_ids from Pass 1, LLM returns:
    • treatment_mean, control_mean, delta for every (arm × outcome) pair
    • sample sizes
    • metric type

Output: Data/Ground_Truth/study_data.jsonl
  One record per study:
  {
    "seq_id": 103,
    "title": "...",
    "paper_files": [...],
    "design_status": "ok|partial|failed",
    "results_status": "ok|partial|failed",
    "instrument": {
      "preamble": "...",
      "control_arm_id": "...",
      "is_simulatable": true,
      "simulatability_note": "...",
      "treatment_variations": [
        {"arm_id": "...", "arm_label": "...", "text": "..."}
      ],
      "outcome_questions": [
        {
          "outcome_id": "...",
          "outcome_name": "...",
          "question_text": "...",
          "response_instruction": "Reply with a single integer from 1 to 7.",
          "scale_type": "likert|binary|continuous|categorical|other",
          "scale_min": 1,
          "scale_max": 7
        }
      ]
    },
    "ground_truth": {
      "effects": [
        {
          "arm_id": "...",
          "outcome_id": "...",
          "outcome_name": "...",
          "delta": 0.45,
          "treatment_mean": 4.1,
          "control_mean": 3.65,
          "n_treatment": 150,
          "n_control": 148,
          "metric": "mean|proportion|coefficient|other",
          "note": "..."
        }
      ]
    }
  }

This output feeds directly into:
  simulate_aggregate.py  — reads instrument
  compare_effects.py     — reads ground_truth.effects

arm_ids and outcome_ids are shared between both sections,
so no matching step is needed at comparison time.

Usage:
    python extract_study_data.py [--model gpt-4.1] [--force]
                                 [--seq-ids 103 150 178] [--pass1-only]
"""

import argparse, asyncio, json, re
from pathlib import Path

import pdfplumber
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR.parents[1] / "Data"
PAPERS_ROOT  = DATA_DIR / "Papers"
OUTPUT_PATH  = DATA_DIR / "Ground_Truth" / "study_data.jsonl"
CACHE_PATH   = DATA_DIR / "Caches" / ".extract_study_data_cache.json"

MAX_PDF_CHARS  = 90_000
MAX_CONCURRENT = 4
DEFAULT_MODEL  = "gpt-4.1"

COVERAGE_OK      = 0.75
COVERAGE_PARTIAL = 0.25

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model",     default=DEFAULT_MODEL)
parser.add_argument("--force",     action="store_true",
                    help="Re-extract even if already cached")
parser.add_argument("--seq-ids",   nargs="*", type=int,
                    help="Only process these seq_ids")
parser.add_argument("--pass1-only", action="store_true",
                    help="Only run design extraction, skip results")
args = parser.parse_args()

client = AsyncOpenAI()

# ---------------------------------------------------------------------------
# Slugify — must stay in sync with simulate_aggregate.py
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


def discover_all_seq_ids() -> list[int]:
    seen = set()
    for f in PAPERS_ROOT.rglob("*.pdf"):
        m = re.match(r"^(\d+)", f.stem)
        if m:
            seen.add(int(m.group(1)))
    return sorted(seen)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a precise research assistant that extracts information from "
    "social science papers. Return only valid JSON with no prose or markdown."
)


def pass1_prompt(paper_text: str) -> str:
    return f"""Extract the experimental design from this social science paper.

Return a JSON object with the following structure.
Use null for any field you cannot determine.

{{
  "title": "<full paper title>",
  "is_simulatable": <true if an LLM can answer the outcome questions after
                    reading the treatment text; false if the study requires
                    visual stimuli the LLM cannot see, real-time task
                    performance, audio/video, or physical interaction>,
  "simulatability_note": "<brief reason if not simulatable, else empty string>",
  "preamble": "<text shown to ALL participants before any condition-specific
               content — e.g. survey introduction, consent language,
               background context. Empty string if none.>",
  "control_arm_label": "<exact label of the control / baseline condition
                         as written in the paper — e.g. 'Control group',
                         'Placebo', 'No treatment'. null if no single
                         control exists (factorial / all-treatment design).>",
  "treatment_variations": [
    {{
      "arm_label": "<condition label as written in the paper>",
      "is_control": <true for the control/baseline arm, false otherwise>,
      "text": "<exact stimulus text, message, or vignette shown to
               participants in this condition. If conditions differ only
               in one inserted phrase, include the full text with the
               variation clearly visible. Empty string if the condition
               is 'no treatment'.>"
    }}
  ],
  "outcome_questions": [
    {{
      "outcome_name": "<outcome measure name as used in the paper>",
      "question_text": "<exact question or task description shown to
                        participants. null if not shown as a direct
                        question (e.g. behavioural measure).>",
      "scale_type": "likert" | "binary" | "continuous" | "categorical" | "other",
      "scale_min": <number or null>,
      "scale_max": <number or null>,
      "scale_labels": ["<label for min>", "...", "<label for max>"] or null
    }}
  ]
}}

Rules:
- Include ALL experimental arms, including the control.
- treatment_variations must have exactly one entry with is_control=true
  (or none if the design has no single control).
- outcome_questions should list only the PRIMARY dependent variables
  used to evaluate treatment effects, not manipulation checks or
  demographic questions.
- If the paper has multiple sub-experiments, extract the MAIN experiment.
- Do not invent content — if stimulus text is not provided verbatim in
  the paper, describe what was shown as precisely as possible.

PAPER TEXT:
{paper_text}"""


def pass2_prompt(paper_text: str, design: dict) -> str:
    ctrl_label = design.get("control_arm_label") or "control"
    arms       = design.get("treatment_variations", [])
    outcomes   = design.get("outcome_questions", [])

    # Build lists using the arm_ids and outcome_ids we will generate
    treatment_arms = [a for a in arms if not a.get("is_control")]
    arm_lines      = "\n".join(
        f'  arm_id="{slugify(a["arm_label"])}"  label="{a["arm_label"]}"'
        for a in treatment_arms
    )
    outcome_lines  = "\n".join(
        f'  outcome_id="{slugify(o["outcome_name"])}"  name="{o["outcome_name"]}"'
        for o in outcomes
    )
    expected_n = len(treatment_arms) * len(outcomes)
    ctrl_id    = slugify(ctrl_label) if ctrl_label else "control"

    return f"""You are extracting statistical results from a social science paper.

STUDY DESIGN ALREADY EXTRACTED:
  Control arm: arm_id="{ctrl_id}"  label="{ctrl_label}"

TREATMENT ARMS:
{arm_lines}

OUTCOME MEASURES:
{outcome_lines}

TASK
For each combination of (treatment arm × outcome), find the result
in the paper's tables or text and extract:

  treatment_mean  — mean / proportion / score for the TREATMENT arm
  control_mean    — mean / proportion / score for the CONTROL arm
  delta           — treatment_mean MINUS control_mean (treatment effect)
  n_treatment     — participants in the treatment arm (integer or null)
  n_control       — participants in the control arm  (integer or null)
  metric          — "mean" | "proportion" | "coefficient" | "other"
  note            — table reference or caveat, e.g. "Table 2 col 3",
                    "OLS coefficient, not raw mean", or ""

Rules:
- Use the arm_id and outcome_id strings EXACTLY as listed above.
- If only a regression coefficient is available (not raw means), set
  delta = coefficient, treatment_mean = null, control_mean = null,
  metric = "coefficient".
- If a value genuinely cannot be found, use null — do NOT guess.
- Return exactly {expected_n} effect objects.

Return ONLY valid JSON — no markdown, no prose:
{{
  "effects": [
    {{
      "arm_id": "<arm_id from list above>",
      "outcome_id": "<outcome_id from list above>",
      "outcome_name": "<outcome name>",
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
# API call
# ---------------------------------------------------------------------------

async def call_api(prompt: str, retries: int = 6) -> str | None:
    import random
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=args.model,
                temperature=0,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content
        except RateLimitError:
            wait = min(20 * (2 ** attempt), 120) + random.uniform(0, 5)
            print(f"    Rate limit — {wait:.1f}s (attempt {attempt+1})")
            await asyncio.sleep(wait)
        except APIConnectionError:
            wait = min(5 * (2 ** attempt), 30) + random.uniform(0, 3)
            print(f"    Connection error — {wait:.1f}s")
            await asyncio.sleep(wait)
        except APIError as e:
            wait = min(10 * (2 ** attempt), 60) + random.uniform(0, 5)
            print(f"    API error ({e}) — {wait:.1f}s")
            await asyncio.sleep(wait)
    return None

# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r'\{[\s\S]*\}', cleaned)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None

# ---------------------------------------------------------------------------
# Build final record from Pass 1 + Pass 2 data
# ---------------------------------------------------------------------------

def build_instrument(design: dict) -> dict:
    """Convert Pass 1 LLM output into the instrument schema."""
    arms     = design.get("treatment_variations", [])
    outcomes = design.get("outcome_questions",    [])

    ctrl_label = design.get("control_arm_label")
    ctrl_id    = slugify(ctrl_label) if ctrl_label else None

    treatment_variations = []
    for a in arms:
        label = a.get("arm_label", "")
        aid   = slugify(label)
        treatment_variations.append({
            "arm_id":    aid,
            "arm_label": label,
            "is_control": bool(a.get("is_control")),
            "text":      a.get("text", ""),
        })

    outcome_questions = []
    for o in outcomes:
        name = o.get("outcome_name", "")
        oid  = slugify(name)
        outcome_questions.append({
            "outcome_id":          oid,
            "outcome_name":        name,
            "question_text":       o.get("question_text"),
            "scale_type":          o.get("scale_type", "other"),
            "scale_min":           o.get("scale_min"),
            "scale_max":           o.get("scale_max"),
            "scale_labels":        o.get("scale_labels"),
            "response_instruction": _build_response_instruction(o),
        })

    return {
        "preamble":            design.get("preamble", ""),
        "control_arm_id":      ctrl_id,
        "is_simulatable":      bool(design.get("is_simulatable", True)),
        "simulatability_note": design.get("simulatability_note", ""),
        "treatment_variations":  treatment_variations,
        "outcome_questions":     outcome_questions,
    }


def _build_response_instruction(outcome: dict) -> str:
    stype = outcome.get("scale_type", "")
    lo    = outcome.get("scale_min")
    hi    = outcome.get("scale_max")
    if stype == "binary":
        return "Reply with exactly one of: YES or NO."
    if stype in ("likert", "continuous") and lo is not None and hi is not None:
        return f"Reply with a single integer from {int(lo)} to {int(hi)} only."
    if stype == "categorical":
        labels = outcome.get("scale_labels") or []
        if labels:
            opts = " or ".join(f'"{l}"' for l in labels)
            return f"Reply with exactly one of: {opts}."
    return "Reply with a number only."


def coverage(effects: list[dict], expected_n: int) -> float:
    if expected_n == 0:
        return 1.0
    found = sum(1 for e in effects if e.get("delta") is not None)
    return found / expected_n


def grade(cov: float) -> str:
    if cov >= COVERAGE_OK:
        return "ok"
    elif cov >= COVERAGE_PARTIAL:
        return "partial"
    return "failed"

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
# Per-study extraction
# ---------------------------------------------------------------------------

async def extract_one(seq_id: int, cache: dict,
                      sem: asyncio.Semaphore) -> dict | None:
    pdf_paths = find_pdfs(seq_id)
    if not pdf_paths:
        print(f"  seq={seq_id:>3}  [NO PDF]")
        return None

    paper_text  = extract_pdf_text(pdf_paths)
    paper_files = [p.name for p in pdf_paths]

    # ---- Pass 1: design ------------------------------------------------
    p1_key = f"p1__{seq_id}__{args.model}"
    if not args.force and p1_key in cache:
        design = cache[p1_key]
        print(f"  seq={seq_id:>3}  [P1 CACHED]", end="")
    else:
        async with sem:
            print(f"  seq={seq_id:>3}  pass 1 (design)…", end="", flush=True)
            raw = await call_api(pass1_prompt(paper_text))
        design = parse_json(raw)
        if design is None:
            print(f"  seq={seq_id:>3}  FAILED (pass 1 parse error)")
            return _failed_record(seq_id, paper_files)
        cache[p1_key] = design
        save_cache(cache)

    n_arms     = len(design.get("treatment_variations", []))
    n_outcomes = len(design.get("outcome_questions", []))
    ctrl_label = design.get("control_arm_label") or "none"
    simulatable = design.get("is_simulatable", True)
    print(f"  arms={n_arms}  outcomes={n_outcomes}  ctrl='{ctrl_label}'  "
          f"simulatable={simulatable}")

    instrument = build_instrument(design)

    # ---- Pass 2: results -----------------------------------------------
    if args.pass1_only:
        return _record(seq_id, design.get("title", ""), paper_files,
                       "ok", "skipped", instrument, [])

    ctrl_id    = instrument.get("control_arm_id")
    treat_arms = [a for a in instrument["treatment_variations"]
                  if not a["is_control"]]
    expected_n = len(treat_arms) * n_outcomes

    if ctrl_id is None or expected_n == 0:
        print(f"  seq={seq_id:>3}  pass 2 skipped "
              f"(no control arm or no outcomes)")
        return _record(seq_id, design.get("title", ""), paper_files,
                       "ok", "skipped", instrument, [])

    p2_key = f"p2__{seq_id}__{args.model}"
    if not args.force and p2_key in cache:
        effects_raw = cache[p2_key]
        n_found = sum(1 for e in effects_raw if e.get("delta") is not None)
        cov = coverage(effects_raw, expected_n)
        print(f"  seq={seq_id:>3}  [P2 CACHED]  "
              f"{grade(cov)}  ({n_found}/{expected_n} deltas)")
    else:
        async with sem:
            print(f"  seq={seq_id:>3}  pass 2 (results, "
                  f"{expected_n} expected)…", end="", flush=True)
            raw2 = await call_api(pass2_prompt(paper_text, design))
        parsed2  = parse_json(raw2)
        effects_raw = (parsed2 or {}).get("effects", [])

        # Retry if coverage is low
        if coverage(effects_raw, expected_n) < COVERAGE_PARTIAL and expected_n > 0:
            print(f"  seq={seq_id:>3}  low coverage — retrying…", end="", flush=True)
            async with sem:
                raw2b = await call_api(pass2_prompt(paper_text, design))
            parsed2b  = parse_json(raw2b)
            effects_b = (parsed2b or {}).get("effects", [])
            if coverage(effects_b, expected_n) > coverage(effects_raw, expected_n):
                effects_raw = effects_b

        n_found = sum(1 for e in effects_raw if e.get("delta") is not None)
        cov = coverage(effects_raw, expected_n)
        print(f"  {grade(cov)}  ({n_found}/{expected_n} deltas)")

        cache[p2_key] = effects_raw
        save_cache(cache)

    results_status = grade(coverage(effects_raw, expected_n))
    return _record(seq_id, design.get("title", ""), paper_files,
                   "ok", results_status, instrument, effects_raw)


def _record(seq_id, title, paper_files, design_status,
            results_status, instrument, effects) -> dict:
    return {
        "seq_id":         seq_id,
        "title":          title,
        "paper_files":    paper_files,
        "design_status":  design_status,
        "results_status": results_status,
        "instrument":     instrument,
        "ground_truth": {
            "effects": effects,
        },
    }


def _failed_record(seq_id, paper_files) -> dict:
    return _record(seq_id, f"seq={seq_id}", paper_files,
                   "failed", "failed", {}, [])

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    seq_ids = args.seq_ids or discover_all_seq_ids()
    print(f"Processing {len(seq_ids)} studies  |  model={args.model}  "
          f"force={args.force}  pass1_only={args.pass1_only}\n")

    cache = load_cache()
    sem   = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [extract_one(sid, cache, sem) for sid in sorted(seq_ids)]
    raw_results = await asyncio.gather(*tasks)
    results     = [r for r in raw_results if r is not None]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for rec in sorted(results, key=lambda r: r["seq_id"]):
            f.write(json.dumps(rec) + "\n")

    ok_d  = sum(1 for r in results if r["design_status"]  == "ok")
    ok_r  = sum(1 for r in results if r["results_status"] == "ok")
    part_r = sum(1 for r in results if r["results_status"] == "partial")
    sim   = sum(1 for r in results
                if r.get("instrument", {}).get("is_simulatable", False))
    total_deltas = sum(
        sum(1 for e in r["ground_truth"]["effects"] if e.get("delta") is not None)
        for r in results
    )

    print(f"\n{'='*60}")
    print(f"Studies processed    : {len(results)}")
    print(f"Design extracted     : {ok_d}")
    print(f"Results ok/partial   : {ok_r}/{part_r}")
    print(f"Simulatable          : {sim}")
    print(f"Total non-null deltas: {total_deltas}")
    print(f"Output → {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
