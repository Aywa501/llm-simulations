"""
02b_simulate_178_patch.py  —  Appendix simulation for seq=178 missing outcomes

Simulates the two outcomes added to study_data.jsonl after the main batch was
submitted (durable_goods, non_durable_goods) for both arms of seq=178.

Appends results to:
    Data/Simulation/aggregate_simulation_raw.jsonl

Usage:
    # Generate a batch input file for manual upload (default):
    python 02b_simulate_178_patch.py [--n 50]

    # Submit directly to OpenAI Batch API:
    python 02b_simulate_178_patch.py --submit [--n 50]

    # Download a completed batch:
    python 02b_simulate_178_patch.py --download BATCH_ID

    # Run live via async API (immediate, full price):
    python 02b_simulate_178_patch.py --async [--n 50]
"""

import argparse, asyncio, io, json, re, time, uuid
from pathlib import Path
from openai import AsyncOpenAI, OpenAI, RateLimitError, APIConnectionError, APIError
import importlib.util, random

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "Data"

# ---------------------------------------------------------------------------
# Import shared helpers from 02_simulate.py
# ---------------------------------------------------------------------------

_spec = importlib.util.spec_from_file_location("simulate", SCRIPT_DIR / "02_simulate.py")
_sim  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sim)

MODEL         = _sim.MODEL
MODEL_PARAMS  = _sim.MODEL_PARAMS
SYSTEM_PROMPT = _sim.SYSTEM_PROMPT
build_prompt  = _sim.build_prompt
simulate_one  = _sim.simulate_one
OUTPUT_PATH   = _sim.OUTPUT_PATH
MAX_ASYNC_CONCURRENT = _sim.MAX_ASYNC_CONCURRENT

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--n",        type=int, default=50, dest="n_per_arm")
parser.add_argument("--submit",   action="store_true")
parser.add_argument("--async",    action="store_true", dest="run_async")
parser.add_argument("--download", metavar="BATCH_ID", default=None)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Targeted config: seq=178, durable_goods + non_durable_goods only
# ---------------------------------------------------------------------------

def parse_durable_goods(text: str) -> float | None:
    t = text.strip().upper()
    if "GOOD" in t:       return 1.0
    if "UNCERTAIN" in t or "DEPENDS" in t: return 0.5
    if "BAD" in t:        return 0.0
    return None


def parse_non_durable_goods(text: str) -> float | None:
    t = text.strip().upper()
    if "MORE" in t:  return 1.0
    if "SAME" in t:  return 0.5
    if "LESS" in t:  return 0.0
    return None


# Load seq=178 from study_data.jsonl so preamble / arm texts stay canonical
_all = [json.loads(l) for l in open(DATA_DIR / "Ground_Truth" / "study_data.jsonl")]
_rec = next(r for r in _all if r["seq_id"] == 178)
_inst = _rec["instrument"]

ARMS = {v["arm_id"]: v["text"] for v in _inst["treatment_variations"]}

_oq_map = {oq["outcome_id"]: oq for oq in _inst["outcome_questions"]}

OUTCOMES = [
    {
        "id":              "durable_goods",
        "response_format": "categorical",
        "question":        (
            "\n\n"
            + _oq_map["durable_goods"]["question_text"] + "\n\n"
            + "Reply with exactly one of: Good time to buy, Uncertain; depends, Bad time to buy."
        ),
        "_parser":         parse_durable_goods,
    },
    {
        "id":              "non_durable_goods",
        "response_format": "categorical",
        "question":        (
            "\n\n"
            + _oq_map["non_durable_goods"]["question_text"] + "\n\n"
            + "Reply with exactly one of: Spend more, Spend same, Spend less."
        ),
        "_parser":         parse_non_durable_goods,
    },
]

CONFIG = {
    "seq_id":   178,
    "preamble": _inst["preamble"],
    "arms":     ARMS,
    "outcomes": OUTCOMES,
}

TOTAL_CALLS = len(ARMS) * len(OUTCOMES) * args.n_per_arm

# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def build_requests() -> list[dict]:
    reqs = []
    for arm_id in ARMS:
        for outcome in OUTCOMES:
            prompt = build_prompt(CONFIG, arm_id, outcome)
            for i in range(args.n_per_arm):
                reqs.append({
                    "custom_id": f"178__{arm_id}__{outcome['id']}__{i}",
                    "method":    "POST",
                    "url":       "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "max_completion_tokens": 4096,
                        **MODEL_PARAMS,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                    },
                })
    return reqs


def generate_batch_file():
    batch_dir = DATA_DIR / "Simulation" / "Batch_Input_Patch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    out_file  = batch_dir / "batch_178_patch.jsonl"
    reqs      = build_requests()
    with open(out_file, "w") as f:
        for r in reqs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    size_kb = out_file.stat().st_size / 1024
    print(f"Wrote {len(reqs)} requests  {size_kb:.1f} KB  → {out_file}")
    print(f"model={MODEL}  n={args.n_per_arm}  arms={list(ARMS)}  outcomes={[o['id'] for o in OUTCOMES]}")
    print(f"\nUpload via OpenAI dashboard or:")
    print(f"  openai api files.create -f {out_file} -p batch")
    print(f"\nAfter batch completes, download with:")
    print(f"  python 02b_simulate_178_patch.py --download BATCH_ID")


def submit_batch():
    client = OpenAI()
    reqs   = build_requests()
    content = "\n".join(json.dumps(r) for r in reqs).encode()
    print(f"Uploading {len(content)/1024:.1f} KB …")
    file_obj = client.files.create(file=io.BytesIO(content), purpose="batch")
    batch    = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"batch_id={batch.id}  status={batch.status}")
    print(f"Download: python 02b_simulate_178_patch.py --download {batch.id}")


def download_and_append(batch_id: str):
    client  = OpenAI()
    batch   = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"Batch not completed yet (status={batch.status})")
        return
    content = client.files.content(batch.output_file_id).text
    records = []
    for line in content.splitlines():
        if not line.strip():
            continue
        r      = json.loads(line)
        parts  = r["custom_id"].split("__")
        arm_id = parts[1]
        oid    = parts[2]
        if r.get("error"):
            records.append({
                "seq_id": 178, "arm_id": arm_id, "outcome_id": oid,
                "pid": r["custom_id"], "response": None,
                "value": None, "parse_ok": False, "error": str(r["error"]),
            })
            continue
        text    = (r["response"]["body"]["choices"][0]["message"]["content"] or "").strip()
        outcome = next(o for o in OUTCOMES if o["id"] == oid)
        value   = outcome["_parser"](text)
        records.append({
            "seq_id": 178, "arm_id": arm_id, "outcome_id": oid,
            "pid": r["custom_id"], "response": text,
            "value": value, "parse_ok": value is not None,
        })
    _append_and_report(records)


# ---------------------------------------------------------------------------
# Async mode
# ---------------------------------------------------------------------------

async def run_async():
    client  = AsyncOpenAI()
    sem     = asyncio.Semaphore(MAX_ASYNC_CONCURRENT)
    records = []
    tasks   = []
    for arm_id in ARMS:
        for outcome in OUTCOMES:
            prompt = build_prompt(CONFIG, arm_id, outcome)
            for _ in range(args.n_per_arm):
                tasks.append(
                    simulate_one(client, 178, arm_id, outcome, prompt,
                                 MODEL_PARAMS, sem)
                )
    print(f"Running {len(tasks)} calls async  (model={MODEL}  n={args.n_per_arm})")
    for coro in asyncio.as_completed(tasks):
        records.append(await coro)
    _append_and_report(records)


# ---------------------------------------------------------------------------
# Append to aggregate_simulation_raw.jsonl and report
# ---------------------------------------------------------------------------

def _append_and_report(records: list[dict]):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_ok   = sum(1 for r in records if r["parse_ok"])
    n_fail = len(records) - n_ok
    print(f"\nAppended {len(records)} records to {OUTPUT_PATH.name}")
    print(f"  parse ok : {n_ok}")
    print(f"  failures : {n_fail}")
    if n_fail:
        for r in records:
            if not r["parse_ok"]:
                print(f"    arm={r['arm_id']}  out={r['outcome_id']}  response={r.get('response')!r}")

    from collections import defaultdict
    sums: dict[tuple, list] = defaultdict(list)
    for r in records:
        if r["parse_ok"] and r["value"] is not None:
            sums[(r["arm_id"], r["outcome_id"])].append(r["value"])
    print("\nArm means:")
    for (arm_id, oid), vals in sorted(sums.items()):
        print(f"  {arm_id:<20}  {oid:<20}  mean={sum(vals)/len(vals):.4f}  n={len(vals)}")

    print("\nNow re-run 04_compare_effects.py and 05_plot.py.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"seq=178 patch  |  outcomes={[o['id'] for o in OUTCOMES]}  "
          f"|  arms={list(ARMS)}  |  n={args.n_per_arm}/arm")

    if args.download:
        download_and_append(args.download)
    elif args.submit:
        submit_batch()
    elif args.run_async:
        asyncio.run(run_async())
    else:
        generate_batch_file()
