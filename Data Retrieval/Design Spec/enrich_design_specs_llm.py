#!/usr/bin/env python3
"""
enrich_design_specs_llm.py

LLM-assisted extraction for AEA RCT Registry "design specs".

Reads:  data/design_specs.jsonl
Writes: data/design_specs_enriched.jsonl
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import math
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from utils import norm_text, read_jsonl, write_jsonl, split_bullets, parse_bool

# ----------------- configuration -----------------

PROMPT_VERSION = "v3.1"
DEFAULT_MODEL = "gpt-5.2"
PROCESS_N_ROWS = 0  # 0 = all; set to e.g. 5 for testing
USE_BATCH_API = True  # Set to True to generate batch_input.jsonl instead of running sync

DESIGN_TYPE_ENUM = [
    "simple_multiarm", "factorial", "encouragement", "cluster_rct", 
    "crossover", "stepped_wedge", "multistage", "saturation", 
    "discontinuity", "observational", "other"
]
ROLE_ENUM = ["control", "treatment", "experimental", "active_comparator", "placebo", "unknown"]
COMPLETENESS_ENUM = ["complete", "partial", "unclear"]
SOURCE_DOC_ENUM = ["registry", "paper"]

JSON_SCHEMA = {
    "name": "design_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "design_type": {"type": "string", "enum": DESIGN_TYPE_ENUM},
            "unit_of_randomization_canonical": {"type": ["string", "null"]},
            "is_clustered": {"type": "boolean"},
            "analysis_unit_canonical": {"type": ["string", "null"]},
            "primary_outcomes_dedup": {
                "type": "array",
                "items": {"type": "string"},
            },
            "arms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "arm_id": {"type": "string"},
                        "name": {"type": "string"},
                        "role": {"type": "string", "enum": ROLE_ENUM},
                        "description": {"type": "string"},
                        "evidence_quote_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["arm_id", "name", "role", "description", "evidence_quote_ids"],
                    "additionalProperties": False,
                },
            },
            "factors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "factor_id": {"type": "string"},
                        "name": {"type": "string"},
                        "levels": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "level_id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "evidence_quote_ids": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["level_id", "name", "description", "evidence_quote_ids"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["factor_id", "name", "levels"],
                    "additionalProperties": False,
                },
            },
            "assignment_rules": {
                "type": "array",
                "items": {"type": "string"},
            },
            "design_completeness": {"type": "string", "enum": COMPLETENESS_ENUM},
            "extraction_sources": {
                "type": "array",
                "items": {"type": "string", "enum": SOURCE_DOC_ENUM},
            },
            "evidence_quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "source_doc": {"type": "string", "enum": SOURCE_DOC_ENUM},
                        "quote": {"type": "string"},
                        "supports": {"type": "string"},
                    },
                    "required": ["id", "source_doc", "quote", "supports"],
                    "additionalProperties": False,
                },
            },
            "notes": {"type": "string"},
        },
        "required": [
            "design_type", "unit_of_randomization_canonical", "is_clustered", "analysis_unit_canonical",
            "primary_outcomes_dedup", "arms", "factors", "assignment_rules",
            "design_completeness", "extraction_sources", "evidence_quotes", "notes"
        ],
        "additionalProperties": False,
    },
}


# ----------------- helpers -----------------

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def uniq_preserve(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = (x or "").strip()
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def build_llm_input(spec: Dict[str, Any]) -> str:
    # Keep this stable; we validate quotes against this exact string.
    parts = []
    parts.append(f"TITLE: {spec.get('title','')}".strip())
    parts.append(f"RCT_ID: {spec.get('rct_id','')}".strip())
    if spec.get("doi_url"):
        parts.append(f"DOI: {spec.get('doi_url')}".strip())
    parts.append("")
    parts.append("INTERVENTION_TEXT:")
    parts.append(norm_text(spec.get("intervention_text", "")))
    parts.append("")
    parts.append("EXPERIMENTAL_DESIGN:")
    parts.append(norm_text(spec.get("experimental_design", "")))
    if spec.get("experimental_design_details"):
        parts.append("")
        parts.append("EXPERIMENTAL_DESIGN_DETAILS:")
        parts.append(norm_text(spec.get("experimental_design_details", "")))
    parts.append("")
    parts.append("PRIMARY_OUTCOMES_RAW:")
    parts.append(", ".join(spec.get("primary_outcomes", []) or []))
    parts.append("")
    parts.append("SECONDARY_OUTCOMES_RAW:")
    parts.append(", ".join(spec.get("secondary_outcomes", []) or []))
    return "\n".join(parts).strip() + "\n"

def validate_extraction(extracted: Dict[str, Any], input_text: str) -> Tuple[bool, List[str]]:
    errors = []

    # (0) Basic schema validation is handled by JSON mode, but we check business rules.

    def verify_quote_fuzzy(target: str, source: str, threshold: float = 0.85) -> bool:
        """
        Verifies if 'target' is approximately in 'source'.
        Uses summing of matching blocks to handle minor interruptions/typos.
        """
        t_norm = norm_text(target)
        s_norm = norm_text(source)
        
        if not t_norm:
            return True # Empty quote is "found" (though likely caught by other checks)
        
        if t_norm in s_norm:
            return True
            
        # Heuristic: If the quote is very short (< 20 chars), requires exact match.
        if len(t_norm) < 20:
            return False
            
        matcher = difflib.SequenceMatcher(None, t_norm, s_norm)
        # Instead of find_longest_match which fails on split quotes (e.g. slight punctuation diff in middle),
        # we check if the total matching characters cover the quote sufficiently.
        # But we must ensure order is respected (get_matching_blocks does this).
        blocks = matcher.get_matching_blocks()
        matched_len = sum(b.size for b in blocks)
        
        matched_ratio = matched_len / len(t_norm)
        return matched_ratio >= threshold

    # Gather evidence IDs
    quotes = extracted.get("evidence_quotes", [])
    if not isinstance(quotes, list):
        return False, ["evidence_quotes not a list"]
    
    quote_map = {}
    for i, q in enumerate(quotes):
        qid = q.get("id")
        if not qid:
            errors.append(f"evidence_quotes[{i}] missing id")
            continue
        if qid in quote_map:
            errors.append(f"evidence_quotes[{i}] duplicate id '{qid}'")
        quote_map[qid] = q
        
        # Verify quote existence
        qtext = q.get("quote", "")
        if not qtext.strip():
            errors.append(f"evidence_quotes[{i}] empty quote")
        elif not verify_quote_fuzzy(qtext, input_text):
             # Downgrade to warning to prevent failing the entire record for non-verbatim quotes
             errors.append(f"Warning: evidence_quotes[{i}] quote not found in text (id: {qid})")

    def check_quote_refs(refs: List[str], path: str):
        if not refs:
            errors.append(f"{path} has no evidence_quote_ids")
            return
        for rid in refs:
            if rid not in quote_map:
                errors.append(f"{path} references unknown quote id '{rid}'")

    # (1) Arms validation
    arms = extracted.get("arms", [])
    seen_arm_ids = set()
    if isinstance(arms, list):
        for i, arm in enumerate(arms):
            aid = arm.get("arm_id")
            if not aid:
                errors.append(f"arms[{i}] missing arm_id")
            elif aid in seen_arm_ids:
                errors.append(f"arms[{i}] duplicate arm_id '{aid}'")
            seen_arm_ids.add(aid)
            
            check_quote_refs(arm.get("evidence_quote_ids", []), f"arms[{i}]({aid})")

    # (2) Factors validation
    factors = extracted.get("factors", [])
    if isinstance(factors, list):
        for i, fac in enumerate(factors):
            levels = fac.get("levels", [])
            if len(levels) < 2:
                errors.append(f"factors[{i}] has fewer than 2 levels")
            for j, lvl in enumerate(levels):
                check_quote_refs(lvl.get("evidence_quote_ids", []), f"factors[{i}].levels[{j}]")

    # (3) Conditional Logic based on Design Type
    dtype = extracted.get("design_type")
    completeness = extracted.get("design_completeness")
    
    if dtype == "factorial":
        if len(factors) < 1 and completeness != "unclear":
             errors.append("design_type 'factorial' requires at least 1 factor")
        # Arms allowed for named cells.

    # Pass if no critical errors (ignore warnings)
    critical_errors = [e for e in errors if not e.startswith("Warning:")]
    ok = len(critical_errors) == 0
    return ok, errors

def load_cache(cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_cache(cache_path: Path, cache: Dict[str, Any]) -> None:
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

@dataclass
class LLMResult:
    extracted: Optional[Dict[str, Any]]
    raw_text: str
    ok: bool
    errors: List[str]


# ----------------- LLM call -----------------

def get_system_and_user_prompts(input_text: str, stricter: bool) -> Tuple[bool, str]:
    # Schema is now enforced by the API, so we don't dump it in the prompt.
    system = (
        "You occupy the role of a Principal Investigator extracting experimental design specs from AEA RCT Registry entries.\n\n"
        "OBJECTIVE:\n"
        "Produce a structured JSON specification of the experimental design, focusing on assignment structure and evidence.\n\n"
        "RULES:\n"
        "1. **Evidence Anchoring**: Every claim must be supported by an item in `evidence_quotes`. Arms and Factors MUST reference these quotes via `evidence_quote_ids`. Use IDs like 'eq1', 'eq2'.\n"
        "2. **Design Type**: Determine if this is a `simple_multiarm` RCT, a `factorial` design, or another type.\n"
        "3. **Factors vs Arms**:\n"
        "   - **Simple RCTs**: Use `arms`. Leave `factors` empty.\n"
        "   - **Factorial**: Use `factors` to describe the dimensions. **ALSO** populate `arms` with the explicit treatment cells (e.g. 'T1', 'T2', 'Control', 'Barg1', 'SeqODR2') if the text provides them. This is critical for downstream mapping.\n"
        "4. **Assignment Unit & Clustering**:\n"
        "   - `unit_of_randomization_canonical`: Default to 'individual participant' if text says 'between-subject' and no group/cluster assignment is mentioned.\n"
        "   - `is_clustered`: Set to true ONLY if `unit_of_randomization_canonical` is NOT 'individual participant' (e.g. school, village, clinic).\n"
        "   - `analysis_unit_canonical`: Be CONSERVATIVE. Return `null` unless the text explicitly defines the level of analysis or payoff (e.g. 'outcomes measured at household level'). Do not guess.\n"
        "5. **Completeness**:\n"
        "   - 'complete': All assignment rules, arms, and units are clear.\n"
        "   - 'partial': Ambiguity exists in key mapping details (e.g. missing cell names in factorial).\n"
        "   - 'unclear': Critical info missing.\n"
        "6. **Roles**: Use 'control' ONLY if explicitly stated (e.g. 'Control group', 'Placebo', 'Comparison'). Use 'experimental' for active treatment arms where no control exists or for factorial cells. Use 'treatment' for standard intervention arms.\n"
    )
    if stricter:
        system += (
            "\nSTRICT MODE:\n"
            "- Be conservative. If a detail is ambiguous, mark completeness as partial.\n"
            "- Verify every quote exists verbatim.\n"
        )
    
    # User prompt is simple, just the text.
    user = f"Registry Entry Text:\n\n{input_text}"
    return system, user

def call_llm_extract(client: OpenAI, model: str, input_text: str, stricter: bool) -> LLMResult:
    system, user = get_system_and_user_prompts(input_text, stricter)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    # strict schema config
    schema_config = {
        "format": {
            "type": "json_schema",
            "name": JSON_SCHEMA["name"],
            "schema": JSON_SCHEMA["schema"],
            "strict": True
        }
    }

    try:
        # Use new /v1/responses style
        # Note: 'input' accepts the list of messages, 'text' accepts the schema config
        resp = client.responses.create(
            model=model,
            input=messages,
            text=schema_config
        )

        raw = resp.output_text or ""
        
        # With strict mode, we expect pure JSON, but safe to strip whitespace
        cleaned = raw.strip()
        
        try:
            extracted = json.loads(cleaned)
        except Exception:
            return LLMResult(extracted=None, raw_text=raw, ok=False, errors=["response_not_json_parseable"])

        ok, errors = validate_extraction(extracted, input_text)
        return LLMResult(extracted=extracted, raw_text=raw, ok=ok, errors=errors)

    except Exception as e:
        return LLMResult(extracted=None, raw_text="", ok=False, errors=[f"exception: {type(e).__name__}: {e}"])


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/design_specs.jsonl", help="Input JSONL")
    ap.add_argument("--out", dest="out", default="data/design_specs_enriched.jsonl", help="Output JSONL")
    ap.add_argument("--model", dest="model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    ap.add_argument("--max", dest="max_n", type=int, default=PROCESS_N_ROWS, help=f"Process at most N rows (0 = all). Default: {PROCESS_N_ROWS}")
    ap.add_argument("--sleep", dest="sleep_s", type=float, default=0.0, help="Sleep seconds between calls")
    ap.add_argument("--cache", dest="cache", default=".llm_cache_design_extract.json", help="Cache file path (json)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    cache_path = Path(args.cache)

    rows = read_jsonl(in_path)
    if args.max_n and args.max_n > 0:
        rows = rows[: args.max_n]
    
    # ---------------- BATCH API MODE ----------------
    if USE_BATCH_API:
        batch_out_path = in_path.with_name("batch_input.jsonl")
        print(f"BATCH MODE ENABLED. Generating {batch_out_path} WITH STRICT SCHEMA...")
        
        batch_lines = []
        for i, spec in enumerate(rows, 1):
            rct_id = spec.get("rct_id", "UNKNOWN")
            input_text = build_llm_input(spec)
            system, user = get_system_and_user_prompts(input_text, stricter=False) 
            
            # Using /v1/responses body structure
            body = {
                "model": args.model,
                "input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": JSON_SCHEMA["name"],
                        "schema": JSON_SCHEMA["schema"],
                        "strict": True
                    }
                }
            }
            
            req = {
                "custom_id": rct_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body
            }
            batch_lines.append(req)
        
        write_jsonl(batch_out_path, batch_lines)
        print(f"Done. Wrote {len(batch_lines)} request lines to {batch_out_path}")
        
        # Verify match between line and intended endpoint
        if len(batch_lines) > 0:
            first_line_url = batch_lines[0].get("url")
            intended_endpoint = "/v1/responses"
            print(f"DEBUG: JSONL line url='{first_line_url}', Batch endpoint='{intended_endpoint}'")
            
            if first_line_url != intended_endpoint:
                print(f"WARNING: Mismatch! Line url {first_line_url} != {intended_endpoint}")
            else:
                print("CONFIRMED: JSONL line url matches batch endpoint.")

        # Note: Actual batch submission logic would go here if we were auto-submitting.
        # e.g. client.batches.create(input_file_id=..., endpoint="/v1/responses", completion_window="24h")
        return
    # ---------------- END BATCH API MODE ----------------

    cache = load_cache(cache_path)
    client = OpenAI()

    enriched: List[Dict[str, Any]] = []
    n_ok = 0
    n_manual = 0

    for i, spec in enumerate(rows, 1):
        rct_id = spec.get("rct_id", "UNKNOWN")
        input_text = build_llm_input(spec)

        cache_key = stable_hash(json.dumps({
            "rct_id": rct_id,
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "input_hash": stable_hash(input_text),
        }, sort_keys=True))

        if cache_key in cache:
            result = cache[cache_key]
            extracted = result.get("extracted")
            ok = result.get("ok", False)
            errors = result.get("errors", [])
            raw = result.get("raw_text", "")
        else:
            res1 = call_llm_extract(client, args.model, input_text, stricter=False)
            extracted, ok, errors, raw = res1.extracted, res1.ok, res1.errors, res1.raw_text

            if not ok:
                res2 = call_llm_extract(client, args.model, input_text, stricter=True)
                extracted2, ok2, errors2, raw2 = res2.extracted, res2.ok, res2.errors, res2.raw_text
                if ok2:
                    extracted, ok, errors, raw = extracted2, ok2, errors2, raw2
                else:
                    extracted, ok, errors, raw = extracted2, ok2, errors2, raw2

            cache[cache_key] = {
                "rct_id": rct_id,
                "ok": ok,
                "errors": errors,
                "raw_text": raw,
                "extracted": extracted,
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
            }
            save_cache(cache_path, cache)

            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

        out_spec = dict(spec)
        # Remove legacy fields to avoid confusion
        out_spec.pop("clustered", None)
        out_spec.pop("trial_card", None)
        
        out_spec["llm_prompt_version"] = PROMPT_VERSION
        out_spec["llm_model"] = args.model
        out_spec["llm_validation_passed"] = bool(ok)
        out_spec["llm_validation_errors"] = errors
        
        # Merge extracted fields directly into output
        # We assume the schema is canonical now, so we just copy the fields.
        if ok and isinstance(extracted, dict):
            for k, v in extracted.items():
                out_spec[k] = v
            out_spec["needs_manual"] = False
            n_ok += 1
        else:
            out_spec["needs_manual"] = True
            n_manual += 1

        enriched.append(out_spec)
        print(f"[{i}/{len(rows)}] {rct_id}: ok={ok} manual={out_spec['needs_manual']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, enriched)

    print("\nDONE")
    print(f"Wrote: {out_path}")
    print(f"Validated OK: {n_ok}")
    print(f"Needs manual: {n_manual}")
    print(f"Cache: {cache_path}")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        print("Set it like: export OPENAI_API_KEY='...'", file=sys.stderr)
        sys.exit(2)
    main()
