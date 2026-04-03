"""
extract_gt_from_papers.py

Fresh ground-truth extraction from PDFs, keyed to the EXACT arm_ids and
outcome_names in simulatable_studies.json — the same IDs the simulator uses.

This replaces the old two-step pipeline (independent GT extraction +
id_mapping.json reconciliation) which suffered from pervasive ID mismatches.
Because this script uses simulatable_studies.json as the schema for what to
extract, compare_aggregate.py can look up values directly with no mapping.

Output:
    Data/Ground_Truth/study_gt_v2.jsonl

    Same structure as study_enriched_aggregate_pass2.jsonl so compare_aggregate.py
    works unchanged (set --gt-file study_gt_v2.jsonl).

    extract_status field:
        "ok"      — ≥80% of arm×outcome pairs have non-null values
        "partial" — some pairs populated but <80%
        "failed"  — no values extracted at all

Validation and retry:
    After the first extraction pass the script checks coverage.
    If any arm×outcome pairs are missing it runs a targeted second pass
    showing only the missing combinations back to the model, asking it to
    look harder (tables, appendices, supplementary material).

Usage:
    python extract_gt_from_papers.py              # all 21 studies, skip cached
    python extract_gt_from_papers.py --study 174  # single study
    python extract_gt_from_papers.py --force      # re-run even if cached
    python extract_gt_from_papers.py --dry-run    # print prompts, no API calls
"""

import argparse, json, random, re, time
from pathlib import Path
from openai import OpenAI, RateLimitError, APIConnectionError, APIError

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR.parents[1] / "Data"
STUDIES_PATH = DATA_DIR / "simulatable_studies.json"
GT_OUT       = DATA_DIR / "Ground_Truth" / "study_gt_v2.jsonl"
PAPERS_ROOT  = DATA_DIR / "Papers"
CACHE_PATH   = DATA_DIR / ".extract_gt_v2_cache.json"

MODEL         = "gpt-5.1"
MAX_PDF_CHARS = 80_000

COVERAGE_OK      = 0.80   # fraction of pairs needed for status="ok"
COVERAGE_PARTIAL = 0.10   # below this → status="failed"

# Cache stores keys as "arm_id|||outcome_name" (JSON-safe)
CACHE_SEP = "|||"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50]


def find_pdfs(seq_id: int) -> list[Path]:
    """All non-supplement PDFs for this study across all person-subdirs."""
    pdfs = []
    for person_dir in sorted(PAPERS_ROOT.iterdir()):
        if not person_dir.is_dir():
            continue
        for p in sorted(person_dir.iterdir()):
            if "supplement" in p.stem.lower():
                continue
            if re.fullmatch(rf"{seq_id}(_\d+)?", p.stem):
                pdfs.append(p)
    return pdfs


def extract_pdf_text(pdfs: list[Path]) -> str:
    import pdfplumber
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
    return "\n\n".join(parts)[:MAX_PDF_CHARS]


# ---------------------------------------------------------------------------
# Cache I/O  (keys stored as "arm_id|||outcome_name" strings)
# ---------------------------------------------------------------------------

def load_cache() -> dict[str, dict]:
    """Returns {str(seq_id): {"arm|||out": {value,...}, ...}}"""
    try:
        return json.loads(CACHE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict):
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def cache_to_values(raw: dict) -> dict[tuple, dict]:
    """Convert cached string-keyed dict → tuple-keyed values dict."""
    return {
        tuple(k.split(CACHE_SEP, 1)): v
        for k, v in raw.items()
    }


def values_to_cache(values: dict[tuple, dict]) -> dict:
    """Convert tuple-keyed values dict → string-keyed dict for JSON storage."""
    return {
        f"{arm_id}{CACHE_SEP}{out_name}": v
        for (arm_id, out_name), v in values.items()
    }


# ---------------------------------------------------------------------------
# Load studies
# ---------------------------------------------------------------------------

def load_studies() -> dict[int, dict]:
    with open(STUDIES_PATH) as f:
        studies = json.load(f)
    return {s["seq_id"]: s for s in studies}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are a careful research assistant extracting numerical results "
    "from a published social science experiment. "
    "Return only valid JSON. Never fabricate values — use null if a value "
    "is not explicitly stated in the paper."
)


def _format_hint(fmt: str, lo, hi) -> str:
    if fmt == "binary":
        return "binary proportion (0=No/0%, 1=Yes/100%)"
    if fmt == "percent":
        return "percentage 0–100"
    if fmt == "proportion":
        return "proportion 0–1"
    if fmt in ("scale", "dollar") and lo is not None and hi is not None:
        return f"scale {lo}–{hi}"
    return "numeric"


def build_first_pass_prompt(study: dict, pdf_text: str) -> str:
    arms  = study.get("treatment_variations", [])
    outs  = study.get("outcome_questions", [])
    title = study.get("title", f"seq={study['seq_id']}")

    arm_id_list  = [v["arm_id"]        for v in arms]
    out_name_list = [q["outcome_name"] for q in outs]

    arms_block = "\n\n".join(
        f'  arm_id: "{v["arm_id"]}"\n'
        f'  Label : {v.get("arm_label", v["arm_id"])}\n'
        f'  Treatment description:\n'
        f'    {v.get("text", "(no description)")[:500]}'
        for v in arms
    )
    outs_block = "\n\n".join(
        f'  outcome_name: "{q["outcome_name"]}"\n'
        f'  Measures    : {q.get("question_text", q.get("response_instruction", ""))[:300]}\n'
        f'  Value format: {_format_hint(q.get("response_format", ""), q.get("scale_min"), q.get("scale_max"))}'
        for q in outs
    )

    return f"""Study: "{title}"

════ EXPERIMENTAL ARMS ════
{arms_block}

════ OUTCOMES TO EXTRACT ════
{outs_block}

════ PAPER TEXT (first {len(pdf_text):,} chars) ════
{pdf_text}
════ END PAPER TEXT ════

TASK:
For every combination of ARM × OUTCOME listed above, find the value
reported in the paper (mean, proportion, or rate as appropriate).

STRICT RULES:
1. arm_id in your response MUST be one of: {arm_id_list}
2. outcome_name in your response MUST be one of: {out_name_list}
3. Report only values EXPLICITLY stated in the paper.
4. Do NOT back-calculate from t-statistics, F-statistics, or p-values.
5. If a value is not reported, set "value": null — do not guess.
6. For every non-null value include a short verbatim quote as "evidence".
7. Include n_analyzed, sd, se if stated; null otherwise.

Return ONLY this JSON (no markdown, no extra keys):
{{
  "arms": [
    {{
      "arm_id": "<one of the arm_ids above>",
      "outcomes": [
        {{
          "outcome_name": "<one of the outcome_names above>",
          "value":      <number | null>,
          "n_analyzed": <number | null>,
          "sd":         <number | null>,
          "se":         <number | null>,
          "evidence":   "<verbatim quote or empty string>"
        }}
      ]
    }}
  ]
}}
"""


def build_retry_prompt(study: dict, pdf_text: str,
                       missing: list[tuple[str, str]]) -> str:
    """Second-pass prompt targeting only the arm×outcome pairs still missing."""
    arm_lookup = {v["arm_id"]: v for v in study.get("treatment_variations", [])}
    out_lookup = {q["outcome_name"]: q for q in study.get("outcome_questions", [])}
    title = study.get("title", f"seq={study['seq_id']}")

    rows = []
    for arm_id, out_name in missing:
        arm = arm_lookup.get(arm_id, {})
        out = out_lookup.get(out_name, {})
        fmt = _format_hint(out.get("response_format", ""),
                           out.get("scale_min"), out.get("scale_max"))
        rows.append(
            f'  arm_id="{arm_id}"  '
            f'label="{arm.get("arm_label", "")}"  '
            f'outcome="{out_name}"  '
            f'format={fmt}'
        )
    missing_block = "\n".join(rows)

    unique_arm_ids  = list(dict.fromkeys(arm_id  for arm_id,  _ in missing))
    unique_out_names = list(dict.fromkeys(out_name for _, out_name in missing))

    return f"""Study: "{title}" — TARGETED SECOND EXTRACTION PASS

The following arm × outcome pairs were NOT found in the first extraction.
These experimental conditions ARE present in the paper — search more
carefully including ALL tables, figures, footnotes, appendices, and
online supplements.

MISSING PAIRS:
{missing_block}

════ PAPER TEXT (first {len(pdf_text):,} chars) ════
{pdf_text}
════ END PAPER TEXT ════

RULES:
- arm_id MUST be one of: {unique_arm_ids}
- outcome_name MUST be one of: {unique_out_names}
- Only report values explicitly stated — if genuinely absent set value: null

Return ONLY this JSON (no markdown):
{{
  "arms": [
    {{
      "arm_id": "<exact arm_id>",
      "outcomes": [
        {{
          "outcome_name": "<exact outcome_name>",
          "value":      <number | null>,
          "n_analyzed": <number | null>,
          "sd":         <number | null>,
          "se":         <number | null>,
          "evidence":   "<verbatim quote or empty string>"
        }}
      ]
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# API call with rate-limit handling
# ---------------------------------------------------------------------------

def call_api(client: OpenAI, prompt: str, retries: int = 6) -> dict | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content or "{}"
            return json.loads(raw)
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
# Parse API response
# ---------------------------------------------------------------------------

def parse_response(result: dict,
                   valid_arm_ids: set[str],
                   valid_out_names: set[str]) -> dict[tuple, dict]:
    """
    Returns {(arm_id, outcome_name): {value, n_analyzed, sd, se, evidence}}.
    Silently drops entries with IDs not in the valid sets.
    """
    extracted: dict[tuple, dict] = {}
    for arm_data in (result or {}).get("arms", []):
        arm_id = arm_data.get("arm_id", "")
        if arm_id not in valid_arm_ids:
            continue
        for o in arm_data.get("outcomes", []):
            out_name = o.get("outcome_name", "")
            if out_name not in valid_out_names:
                continue
            extracted[(arm_id, out_name)] = {
                "value":      o.get("value"),
                "n_analyzed": o.get("n_analyzed"),
                "sd":         o.get("sd"),
                "se":         o.get("se"),
                "evidence":   o.get("evidence", ""),
            }
    return extracted


def count_populated(values: dict[tuple, dict],
                    expected: list[tuple]) -> int:
    return sum(
        1 for k in expected
        if k in values and values[k]["value"] is not None
    )


# ---------------------------------------------------------------------------
# Build output GT record
# ---------------------------------------------------------------------------

def build_gt_record(study: dict,
                    values: dict[tuple, dict],
                    cov: float) -> dict:
    """
    Returns a record in the study_enriched_aggregate JSONL schema so
    compare_aggregate.py can consume it directly.
    """
    arms = study.get("treatment_variations", [])
    outs = study.get("outcome_questions", [])

    if cov >= COVERAGE_OK:
        status = "ok"
    elif cov >= COVERAGE_PARTIAL:
        status = "partial"
    else:
        status = "failed"

    outcomes = []
    for q in outs:
        out_name = q["outcome_name"]
        gs_list  = []
        for v in arms:
            arm_id = v["arm_id"]
            entry  = values.get((arm_id, out_name), {})
            gs_list.append({
                "arm_id":     arm_id,
                "arm_label":  v.get("arm_label", arm_id),
                "n_analyzed": entry.get("n_analyzed"),
                "metric":     "mean",
                "value":      entry.get("value"),
                "sd":         entry.get("sd"),
                "se":         entry.get("se"),
                "evidence":   entry.get("evidence", ""),
            })
        outcomes.append({
            "name":            out_name,
            "is_primary":      True,
            "outcome_type":    "continuous",
            "group_summaries": gs_list,
        })

    return {
        "seq_id":         study["seq_id"],
        "title":          study.get("title", f"seq={study['seq_id']}"),
        "extract_status": status,
        "coverage":       round(cov, 3),
        "results": {"outcomes": outcomes},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=int, default=None,
                        help="Run for a single seq_id only")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result is cached")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling the API")
    args = parser.parse_args()

    studies = load_studies()
    cache   = load_cache()
    client  = None if args.dry_run else OpenAI()

    GT_OUT.parent.mkdir(parents=True, exist_ok=True)

    seq_ids = [args.study] if args.study else sorted(studies)

    # Load any existing output records we are NOT re-running
    existing: dict[int, dict] = {}
    if GT_OUT.exists():
        for line in open(GT_OUT):
            rec = json.loads(line)
            existing[rec["seq_id"]] = rec

    new_results: dict[int, dict] = {}

    for seq_id in seq_ids:
        study = studies.get(seq_id)
        if study is None:
            print(f"seq={seq_id:>3}  SKIP  (not in simulatable_studies.json)")
            continue

        arms      = study.get("treatment_variations", [])
        outs      = study.get("outcome_questions", [])
        arm_ids   = {v["arm_id"] for v in arms}
        out_names = {q["outcome_name"] for q in outs}
        expected  = [(v["arm_id"], q["outcome_name"])
                     for v in arms for q in outs]
        n_exp     = len(expected)
        title     = study.get("title", f"seq={seq_id}")

        # ── Cache check ──
        cache_key = str(seq_id)
        if not args.force and cache_key in cache:
            values = cache_to_values(cache[cache_key])
            cov    = count_populated(values, expected) / n_exp if n_exp else 0
            rec    = build_gt_record(study, values, cov)
            new_results[seq_id] = rec
            n_pop = count_populated(values, expected)
            print(f"seq={seq_id:>3}  CACHED  "
                  f"{n_pop:>3}/{n_exp:<3} populated  "
                  f"status={rec['extract_status']}  {title[:45]}")
            continue

        # ── Find PDFs ──
        pdfs = find_pdfs(seq_id)
        if not pdfs:
            print(f"seq={seq_id:>3}  NO PDF FOUND — skipping")
            continue

        print(f"seq={seq_id:>3}  PDFs: {[p.name for p in pdfs]}")

        if args.dry_run:
            print(f"── FIRST-PASS PROMPT (seq={seq_id}) ──")
            print(build_first_pass_prompt(study, "[pdf text]")[:1500], "…\n")
            continue

        pdf_text = extract_pdf_text(pdfs)
        print(f"         PDF: {len(pdf_text):,} chars  |  "
              f"{n_exp} expected pairs  ({len(arms)} arms × {len(outs)} outcomes)")

        # ── Pass 1 ──
        print(f"         Pass 1 …")
        result1 = call_api(client, build_first_pass_prompt(study, pdf_text))
        values  = parse_response(result1, arm_ids, out_names)
        n_pop1  = count_populated(values, expected)
        print(f"         Pass 1: {n_pop1}/{n_exp} populated ({100*n_pop1/n_exp:.0f}%)")

        # ── Pass 2: target missing pairs ──
        missing = [k for k in expected
                   if k not in values or values[k]["value"] is None]
        if missing:
            print(f"         Pass 2: targeting {len(missing)} missing pairs …")
            result2     = call_api(client, build_retry_prompt(study, pdf_text, missing))
            retry_vals  = parse_response(result2, arm_ids, out_names)
            new_fills   = 0
            for k in missing:
                rv = retry_vals.get(k, {})
                if rv.get("value") is not None:
                    values[k] = rv
                    new_fills += 1
            n_pop2 = count_populated(values, expected)
            print(f"         Pass 2: +{new_fills} new  → {n_pop2}/{n_exp} "
                  f"({100*n_pop2/n_exp:.0f}%)")

        # ── Final status ──
        cov = count_populated(values, expected) / n_exp if n_exp else 0
        rec = build_gt_record(study, values, cov)
        new_results[seq_id] = rec

        flag = "✓" if rec["extract_status"] == "ok" else \
               ("⚠" if rec["extract_status"] == "partial" else "✗")
        print(f"         Final: {count_populated(values,expected)}/{n_exp}  "
              f"{flag} {rec['extract_status'].upper()}")

        if rec["extract_status"] == "failed":
            print(f"         *** WARNING: no values extracted for seq={seq_id} ***")
        elif rec["extract_status"] == "partial":
            still_null = [k for k in expected
                          if k not in values or values[k]["value"] is None]
            for arm_id, out_name in still_null[:8]:
                print(f"           still missing: arm={arm_id}  out={out_name}")
            if len(still_null) > 8:
                print(f"           … and {len(still_null)-8} more")

        # ── Save to cache incrementally ──
        cache[cache_key] = values_to_cache(values)
        save_cache(cache)

    # ── Merge new results into existing, write output ──
    if not args.dry_run:
        final = {**existing, **new_results}
        with open(GT_OUT, "w") as f:
            for sid in sorted(final):
                f.write(json.dumps(final[sid], ensure_ascii=False) + "\n")
        print(f"\nWrote {GT_OUT}  ({len(final)} studies)")

    # ── Summary report ──
    all_results = {**existing, **new_results}
    print("\n══ Coverage Summary ══")
    print(f"  {'seq':>4}  {'status':<8}  {'cov':>6}  title")
    counts = {"ok": 0, "partial": 0, "failed": 0}
    for sid in sorted(all_results):
        rec  = all_results[sid]
        st   = rec["extract_status"]
        cov  = rec.get("coverage", 0)
        flag = "✓" if st == "ok" else ("⚠" if st == "partial" else "✗")
        counts[st] = counts.get(st, 0) + 1
        print(f"  {sid:>4}  {st:<8}  {cov:>5.0%}  {flag}  {rec['title'][:55]}")

    print()
    print(f"  ok={counts['ok']}  partial={counts['partial']}  "
          f"failed={counts['failed']}  "
          f"total={sum(counts.values())}")
    print()
    print("Next steps:")
    print("  python simulate_aggregate.py --generate-only "
          "--config no_reasoning --n 50")
    print("  [upload Batch_Input/ → download to Batch_Output/]")
    print("  python unpack_aggregate_batches.py")
    print("  python compare_aggregate.py --config no_reasoning "
          "--gt-file study_gt_v2.jsonl")


if __name__ == "__main__":
    main()
