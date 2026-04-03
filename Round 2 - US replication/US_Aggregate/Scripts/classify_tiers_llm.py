"""
classify_tiers_llm.py

Step 2 of the aggregate pipeline.

Reads Data/expanded_us_pool.jsonl and classifies each study as Tier 1, 2, or 3
using an LLM, based on the registry text (no design spec enrichment needed).

Tier definitions:
  Tier 1 - No role casting: LLM responds as itself. Experiment is a survey or
            vignette that any respondent can answer without adopting a persona
            (e.g., "do you support this policy?", conjoint choice, dictator game).
  Tier 2 - Role casting needed: LLM must adopt a human persona or demographic
            identity to respond meaningfully (e.g., "imagine you are a teacher").
  Tier 3 - Heavy role casting or out of scope: requires LLM to act as an
            institution/organisation, OR is a field/lab experiment whose outcome
            is behavioural performance that cannot be elicited via survey question
            (real-effort tasks, audio/video treatments, children as subjects, etc.).

Results are cached so the script is safe to re-run.

Run from this script's directory:
    python classify_tiers_llm.py [--limit N] [--model MODEL]
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
DATA_DIR  = Path(__file__).resolve().parents[1] / "Data"
INPUT     = DATA_DIR / "Pipeline" / "expanded_us_pool.jsonl"
OUTPUT    = DATA_DIR / "Pipeline" / "tier_classified.jsonl"
CACHE_FILE = DATA_DIR / "Caches" / ".tier_cache.json"

DEFAULT_MODEL   = "gpt-5.4-mini"  # fast + cheap for classification at 192 studies
MAX_CONCURRENT  = 20

SYSTEM_PROMPT = """You are classifying social science RCT studies by how much role-casting
an LLM would need to replicate the experiment.

TIER 1 – No role casting.
The experiment is an online survey or vignette study where participants respond as
themselves. The LLM can answer as a generic respondent with no persona assigned.
Examples: opinion survey, conjoint choice between options, dictator game donation,
stated-preference question, policy support question, information treatment + belief update.

TIER 2 – Role casting needed.
The experiment requires participants to have or adopt a specific demographic identity,
profession, or lived experience for the response to be meaningful.
Examples: experiments specifically recruiting teachers/nurses/immigrants, studies where
the treatment interacts with the participant's own group membership.

TIER 3 – Heavy role casting OR fundamentally out of scope.
Assign Tier 3 if ANY of the following apply:
- Participants must act as an institution, organisation, firm, or government body.
- The outcome is real behavioural performance (data entry errors, physical effort,
  actual purchase behaviour, clinical outcomes, academic test scores).
- The treatment is delivered via audio clip, video, in-person interaction, or physical
  environment (not readable text).
- Subjects are children, patients, or other populations an LLM cannot represent.
- The study is a field experiment where outcomes are observed administrative records
  (insurance take-up, earnings, recidivism) rather than survey responses.

Classify conservatively: if genuinely uncertain between 1 and 2, choose 2.
If genuinely uncertain between 2 and 3, choose 3."""


def make_prompt(rec: dict) -> str:
    parts = [f"TITLE: {rec.get('title', '')}"]
    if rec.get("intervention_text"):
        parts.append(f"INTERVENTION: {rec['intervention_text'][:1200]}")
    if rec.get("experimental_design"):
        parts.append(f"EXPERIMENTAL DESIGN: {rec['experimental_design'][:1200]}")
    if rec.get("experimental_design_details"):
        parts.append(f"DESIGN DETAILS: {rec['experimental_design_details'][:600]}")
    if rec.get("primary_outcomes"):
        parts.append(f"PRIMARY OUTCOMES: {rec['primary_outcomes'][:400]}")
    if rec.get("randomization_unit"):
        parts.append(f"RANDOMIZATION UNIT: {rec['randomization_unit']}")
    return "\n\n".join(parts)


def cache_key(rec: dict) -> str:
    text = make_prompt(rec)
    return hashlib.md5(text.encode()).hexdigest()


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


JSON_INSTRUCTION = """
Respond with a JSON object only — no markdown, no explanation outside the JSON.
Schema:
{
  "tier": 1 | 2 | 3,
  "reason": "<one or two sentences>",
  "simulatable": true | false
}"""


def parse_json_response(text: str) -> dict:
    """Extract JSON from response, stripping any markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    return json.loads(text)


async def classify_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    rec: dict,
    cache: dict,
    model: str,
) -> dict:
    key = cache_key(rec)
    if key in cache:
        result = cache[key]
    else:
        async with sem:
            prompt = make_prompt(rec)
            response = await client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt + JSON_INSTRUCTION},
                ],
            )
        raw = response.choices[0].message.content
        result = parse_json_response(raw)
        cache[key] = result

    return {
        **rec,
        "llm_tier":          result["tier"],
        "llm_tier_reason":   result["reason"],
        "llm_simulatable":   result["simulatable"],
    }


async def run(model: str, limit: int):
    # Load pool
    records = []
    with open(INPUT) as f:
        for line in f:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]
    print(f"Classifying {len(records)} studies with {model} ...")

    cache = load_cache()
    cached_hits = sum(1 for r in records if cache_key(r) in cache)
    print(f"  {cached_hits} already cached, {len(records) - cached_hits} new API calls")

    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [classify_one(client, sem, r, cache, model) for r in records]

    results = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        done += 1
        if done % 10 == 0 or done == len(tasks):
            save_cache(cache)
            print(f"  {done}/{len(tasks)} done", end="\r")

    save_cache(cache)
    print()

    # Sort back to original order
    seq_map = {r["seq_id"]: r for r in results}
    ordered = [seq_map[r["seq_id"]] for r in records if r["seq_id"] in seq_map]

    # Summary
    tier_counts = {1: 0, 2: 0, 3: 0}
    for r in ordered:
        tier_counts[r["llm_tier"]] += 1
    print(f"Tier 1: {tier_counts[1]}  Tier 2: {tier_counts[2]}  Tier 3: {tier_counts[3]}")
    print(f"Simulatable (Tier 1 + some Tier 2): "
          f"{sum(1 for r in ordered if r['llm_simulatable'])}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ordered)} records → {OUTPUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--limit",  type=int, default=0, help="Process first N only (0=all)")
    args = parser.parse_args()
    asyncio.run(run(args.model, args.limit))


if __name__ == "__main__":
    main()
