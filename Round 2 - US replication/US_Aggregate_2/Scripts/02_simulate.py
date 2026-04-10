"""
02_simulate.py  —  Step 2 of the pipeline

LLM simulation: for every study in Data/Ground_Truth/study_data.jsonl that is
marked is_simulatable=true, generates N simulated participant responses per arm
× outcome combination using gpt-4.1 at temperature=1.

Reads  : Data/Ground_Truth/study_data.jsonl
Writes : Data/Simulation/Batch_Input/batch_*.jsonl     (default / --generate)
         Data/Simulation/aggregate_simulation_raw.jsonl (--async or --download)

Usage:
    # Generate batch input file for manual upload (default):
    python 02_simulate.py [--n 50]

    # Submit directly to OpenAI Batch API:
    python 02_simulate.py --submit [--n 50]

    # Download a completed batch:
    python 02_simulate.py --download BATCH_ID

    # Run live via async API (immediate, full price):
    python 02_simulate.py --async [--n 50]

    # List loaded study configs:
    python 02_simulate.py --list
"""

import argparse, asyncio, io, json, random, re, time, uuid
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from openai import AsyncOpenAI, OpenAI, RateLimitError, APIConnectionError, APIError

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR.parent / "Data"
STUDIES_PATH = DATA_DIR / "Ground_Truth" / "study_data.jsonl"
MODEL        = "gpt-4.1"

MAX_TOKENS_PER_CHUNK  = 1_345_000
MAX_ASYNC_CONCURRENT  = 20   # semaphore cap for async mode (avoids rate limit burst)

MODEL_PARAMS = {"temperature": 1, "top_p": 1}

OUTPUT_PATH  = DATA_DIR / "Simulation" / "aggregate_simulation_raw.jsonl"


SYSTEM_PROMPT = (
    "You are a participant in an online survey. "
    "Read the scenario and respond as an ordinary American adult. "
    "IMPORTANT: Give only the exact response format requested — "
    "a single word, a single number, or YES/NO. "
    "No explanations, no caveats, no extra text."
)

# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def universal_categorical_parse(text: str) -> float | None:
    t = text.strip().upper()

    if "MESSAGE 1" in t: return 1.0
    if "MESSAGE 2" in t: return 0.0
    if "INCREASE" in t: return 1.0
    if "DECREASE" in t: return 0.0
    if "NEITHER" in t: return 0.5
    if "INFLATION" in t: return 1.0
    if "UNEMPLOYMENT" in t: return 0.0
    if "FOR" in t.split() or "FOR." in t: return 1.0
    if "AGAINST" in t.split() or "AGAINST." in t: return 0.0

    if "VERY SURE" in t: return 1.0
    if "VERY UNSURE" in t: return 0.0
    if t == "SURE" or "SURE." in t: return 0.75
    if t == "UNSURE" or "UNSURE." in t: return 0.25

    if t == "ALL" or t.startswith("ALL "): return 1.0
    if t == "NONE" or t.startswith("NONE "): return 0.0
    if "SOME" in t.split() or "SOME." in t: return 0.5
    if "DON'T KNOW" in t or "DO NOT KNOW" in t: return 0.5

    if "ALREADY KNEW" in t or "ALREADY CONSIDERED" in t: return 1.0
    if "DIDN'T FIND" in t or "NOT CONVINCING" in t or "UNCONVINCING" in t: return 0.0
    if t == "OTHER" or t.startswith("OTHER"): return 0.5

    if "INTEREST" in t: return 0.5
    if "PRICE" in t: return 1.0
    if "CONSUMPTION" in t: return 0.5
    if "EMPLOYMENT" in t or "LABOR" in t or "LABOUR" in t: return 0.0
    if "INCOME" in t or "WEALTH" in t: return 0.5
    if "STOCK" in t or "HOUSING" in t: return 0.5
    if t in ("C", "D", "E", "F", "G"): return 0.5

    if t.startswith("YES"): return 1.0
    if t.startswith("NO"):  return 0.0
    if re.search(r'\bYES\b', t): return 1.0
    if re.search(r'\bNO\b',  t): return 0.0

    positive = r'\b(SUPPORT|FAVOR|AGREE|TRUE|ACCEPT|APPROVE|WILLING|1|A|OPTION A|JOB A)\b'
    negative = r'\b(OPPOSE|PREFER THE CURRENT|DISAGREE|FALSE|REJECT|DISAPPROVE|UNWILLING|0|B|OPTION B|JOB B)\b'
    if re.search(positive, t): return 1.0
    if re.search(negative, t): return 0.0

    m = re.search(r'^[-+]?\d+\.?\d*$', t)
    if m:
        return float(m.group())

    return None


def parse_binary(text: str) -> float | None:
    return universal_categorical_parse(text)


def parse_percent(text: str) -> float | None:
    m = re.search(r"(\d+\.?\d*)\s*%", text)
    if not m:
        m = re.search(r"[-+]?\d+\.?\d*", text.strip())
    if not m:
        return None
    val = float(m.group(1) if m.lastindex else m.group())
    return max(0.0, min(100.0, val))


def parse_proportion(text: str) -> float | None:
    m = re.search(r"[-+]?\d+\.?\d*", text.strip())
    if not m:
        return None
    val = float(m.group())
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def parse_integer(text: str) -> float | None:
    m = re.search(r"[-+]?\d+", text.strip())
    return float(int(m.group())) if m else None


def parse_choice_ab(text: str) -> float | None:
    val = universal_categorical_parse(text)
    if val is not None:
        return val
    m = re.search(r'[-+]?\d+', text.strip())
    return float(int(m.group())) if m else None


def make_scale_parser(lo: float, hi: float):
    def parser(text: str) -> float | None:
        m = re.search(r"[-+]?\d+\.?\d*", text.strip())
        if m:
            return max(lo, min(hi, float(m.group())))
        val = universal_categorical_parse(text)
        if val is not None:
            return lo + val * (hi - lo)
        return None
    return parser


FIXED_PARSERS = {
    "binary":     parse_binary,
    "percent":    parse_percent,
    "proportion": parse_proportion,
    "integer":    parse_integer,
    "choice":     parse_choice_ab,
    "choice_ab":  parse_choice_ab,
    "other":      universal_categorical_parse,
}


def resolve_parser(fmt: str, scale_min=None, scale_max=None):
    if fmt in FIXED_PARSERS:
        return FIXED_PARSERS[fmt]
    if fmt in ("scale", "dollar") and scale_min is not None and scale_max is not None:
        return make_scale_parser(float(scale_min), float(scale_max))
    def _flexible_parser(text: str) -> float | None:
        val = universal_categorical_parse(text)
        if val is not None:
            return val
        return parse_integer(text)
    return _flexible_parser

# ---------------------------------------------------------------------------
# Slugify — must match 01_extract_study_data.py and 04_compare_effects.py
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50]

# ---------------------------------------------------------------------------
# Load study configs from study_data.jsonl
# ---------------------------------------------------------------------------

def load_study_configs(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run 01_extract_study_data.py first."
        )

    configs = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            seq_id     = rec["seq_id"]
            instrument = rec.get("instrument", {})

            if not instrument:
                print(f"  seq={seq_id}  skipped: no instrument")
                continue
            if instrument.get("is_simulatable") is False:
                note = instrument.get("simulatability_note", "")
                print(f"  seq={seq_id}  skipped: not simulatable  ({note})")
                continue

            variations    = instrument.get("treatment_variations", [])
            out_questions = instrument.get("outcome_questions", [])

            if not variations:
                print(f"  seq={seq_id}  skipped: no treatment_variations")
                continue
            if not out_questions:
                print(f"  seq={seq_id}  skipped: no outcome_questions")
                continue

            arms: dict[str, str] = {}
            for v in variations:
                arm_id       = v.get("arm_id") or slugify(v.get("arm_label", "arm"))
                arms[arm_id] = v.get("text", "")

            outcomes = []
            for q in out_questions:
                fmt = q.get("scale_type", "other")
                lo  = q.get("scale_min")
                hi  = q.get("scale_max")
                oid = q.get("outcome_id") or slugify(q.get("outcome_name", "outcome"))
                qt  = (q.get("question_text") or "").strip()
                ri  = q.get("response_instruction") or "Reply with a number."
                # Include question_text in prompt when available so the LLM
                # knows exactly what outcome to report (not just the response format).
                question_block = "\n\n" + (f"{qt}\n\n{ri}" if qt else ri)
                outcomes.append({
                    "id":              oid,
                    "response_format": fmt,
                    "question":        question_block,
                    "_parser":         resolve_parser(fmt, lo, hi),
                })

            configs[seq_id] = {
                "seq_id":   seq_id,
                "preamble": instrument.get("preamble") or "",
                "arms":     arms,
                "outcomes": outcomes,
            }

    return configs

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(config: dict, arm_id: str, outcome: dict) -> str:
    return config["preamble"] + config["arms"][arm_id] + outcome["question"]

# ---------------------------------------------------------------------------
# Async simulation runner
# ---------------------------------------------------------------------------

async def simulate_one(client, seq_id, arm_id, outcome, prompt, model_params,
                       sem: asyncio.Semaphore,
                       retries: int = 6) -> dict:
    pid        = str(uuid.uuid4())
    last_error = ""
    for attempt in range(retries):
        try:
            async with sem:
                r = await client.chat.completions.create(
                    model=MODEL,
                    max_completion_tokens=4096,
                    **model_params,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                )
            text  = (r.choices[0].message.content or "").strip()
            value = outcome["_parser"](text)
            return {
                "seq_id":     seq_id,
                "arm_id":     arm_id,
                "outcome_id": outcome["id"],
                "pid":        pid,
                "response":   text,
                "value":      value,
                "parse_ok":   value is not None,
            }
        except RateLimitError as e:
            last_error = str(e)
            await asyncio.sleep(min(120, 20 * 2 ** attempt) + random.uniform(0, 5))
        except APIConnectionError as e:
            last_error = str(e)
            await asyncio.sleep(min(30, 5 * 2 ** attempt) + random.uniform(0, 2))
        except APIError as e:
            last_error = str(e)
            await asyncio.sleep(min(60, 10 * 2 ** attempt) + random.uniform(0, 3))
        except Exception as e:
            last_error = str(e)
            break
    return {
        "seq_id": seq_id, "arm_id": arm_id, "outcome_id": outcome["id"],
        "pid": pid, "response": None, "value": None,
        "parse_ok": False, "error": last_error,
    }


async def run_study(client, config, n_per_arm, writer, pbar, model_params, sem):
    tasks = []
    for arm_id in config["arms"]:
        for outcome in config["outcomes"]:
            prompt = build_prompt(config, arm_id, outcome)
            for _ in range(n_per_arm):
                tasks.append(
                    simulate_one(client, config["seq_id"], arm_id,
                                 outcome, prompt, model_params, sem)
                )
    for coro in asyncio.as_completed(tasks):
        rec = await coro
        writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pbar.update(1)


async def run_async(n_per_arm: int, study_configs: dict):
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(MAX_ASYNC_CONCURRENT)
    total  = sum(len(c["arms"]) * len(c["outcomes"]) * n_per_arm
                 for c in study_configs.values())
    print(f"model={MODEL}  temperature=1  |  studies={sorted(study_configs)}  "
          f"|  n={n_per_arm}  |  calls={total}  |  concurrency={MAX_ASYNC_CONCURRENT}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        with tqdm(total=total, unit="call") as pbar:
            for seq_id, config in study_configs.items():
                pbar.set_description(f"seq={seq_id}")
                await run_study(client, config, n_per_arm, f, pbar, MODEL_PARAMS, sem)

    with open(OUTPUT_PATH) as f:
        records = [json.loads(l) for l in f]
    _print_summary(records)
    print(f"Output → {OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# Batch mode helpers
# ---------------------------------------------------------------------------

def build_batch_requests(n_per_arm: int, study_configs: dict) -> list[dict]:
    model_params = MODEL_PARAMS
    requests = []
    for seq_id, config in study_configs.items():
        for arm_id in config["arms"]:
            for outcome in config["outcomes"]:
                prompt = build_prompt(config, arm_id, outcome)
                for i in range(n_per_arm):
                    requests.append({
                        "custom_id": f"{seq_id}__{arm_id}__{outcome['id']}__{i}",
                        "method":    "POST",
                        "url":       "/v1/chat/completions",
                        "body": {
                            "model": MODEL,
                            "max_completion_tokens": 4096,
                            **model_params,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user",   "content": prompt},
                            ],
                        },
                    })
    return requests


def chunk_requests(requests: list[dict],
                   max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[list[dict]]:
    chunks, current, current_tok = [], [], 0
    for r in requests:
        est = len(json.dumps(r)) // 4
        if current_tok + est > max_tokens and current:
            chunks.append(current)
            current, current_tok = [], 0
        current.append(r)
        current_tok += est
    if current:
        chunks.append(current)
    return chunks


def generate_batch_file(n_per_arm: int, study_configs: dict):
    batch_dir = DATA_DIR / "Simulation" / "Batch_Input"
    batch_dir.mkdir(parents=True, exist_ok=True)

    for old in batch_dir.glob("batch_*.jsonl"):
        old.unlink()

    requests = build_batch_requests(n_per_arm, study_configs)
    chunks   = chunk_requests(requests)

    for i, chunk in enumerate(chunks):
        out_file = batch_dir / f"batch_{i+1:02d}_of_{len(chunks):02d}.jsonl"
        with open(out_file, "w") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        size_kb    = out_file.stat().st_size / 1024
        est_tokens = sum(len(json.dumps(r)) for r in chunk) // 4
        print(f"  File {i+1}/{len(chunks)}: {len(chunk):>5} requests  "
              f"{size_kb:>7.1f} KB  ~{est_tokens:>7,} tokens  → {out_file.name}")

    print(f"\nmodel={MODEL}  temperature=1  |  n={n_per_arm}  |  {len(chunks)} file(s)")
    print(f"Upload via OpenAI dashboard or:")
    for f in sorted(batch_dir.glob("batch_*.jsonl")):
        print(f"  openai api files.create -f {f} -p batch")


def submit_batch(n_per_arm: int, study_configs: dict) -> list[str]:
    client   = OpenAI()
    requests = build_batch_requests(n_per_arm, study_configs)
    chunks   = chunk_requests(requests)
    print(f"model={MODEL}  temperature=1  |  {len(requests)} requests  |  {len(chunks)} chunk(s)")

    batch_ids = []
    for i, chunk in enumerate(chunks):
        content  = "\n".join(json.dumps(r) for r in chunk).encode()
        print(f"\n[Chunk {i+1}/{len(chunks)}] Uploading {len(content)/1024:.1f} KB …")
        file_obj = client.files.create(file=io.BytesIO(content), purpose="batch")
        batch    = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  batch_id={batch.id}  status={batch.status}")
        print(f"  Download: python 02_simulate.py --download {batch.id}")
        batch_ids.append(batch.id)
        time.sleep(1)

    return batch_ids


def download_batch(batch_id: str, study_configs: dict):
    client = OpenAI()
    print(f"Checking batch {batch_id} …")
    while True:
        batch  = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  status={batch.status}  "
              f"completed={counts.completed}/{counts.total}  "
              f"failed={counts.failed}")
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch ended with status: {batch.status}")
        time.sleep(30)

    if batch.output_file_id is None:
        if batch.error_file_id:
            err = client.files.content(batch.error_file_id).text
            for s in err.splitlines()[:3]:
                print("  ERROR:", s[:300])
        raise RuntimeError("Batch produced no output.")

    raw     = client.files.content(batch.output_file_id).text
    records = _parse_batch_output(raw, study_configs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _print_summary(records)
    print(f"Output → {OUTPUT_PATH}")


def _parse_batch_output(raw: str, study_configs: dict) -> list[dict]:
    outcome_lookup: dict[tuple, dict] = {}
    for seq_id, config in study_configs.items():
        for outcome in config["outcomes"]:
            for arm_id in config["arms"]:
                outcome_lookup[(seq_id, arm_id.strip("_"),
                                outcome["id"].strip("_"))] = outcome

    records = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        r      = json.loads(line)
        parts  = r["custom_id"].split("__")
        seq_id = int(parts[0])
        out_id = re.sub(r"_dup\d+$", "", parts[-2]).strip("_")
        arm_id = "__".join(parts[1:-2]).strip("_")

        if r.get("error"):
            records.append({
                "seq_id": seq_id, "arm_id": arm_id, "outcome_id": out_id,
                "pid": r["custom_id"], "response": None,
                "value": None, "parse_ok": False,
                "error": str(r["error"]),
            })
            continue

        text    = (r["response"]["body"]["choices"][0]["message"]["content"]
                   or "").strip()
        outcome = outcome_lookup.get((seq_id, arm_id, out_id))
        parser  = outcome["_parser"] if outcome else parse_integer
        value   = parser(text)
        records.append({
            "seq_id": seq_id, "arm_id": arm_id, "outcome_id": out_id,
            "pid": r["custom_id"], "response": text,
            "value": value, "parse_ok": value is not None,
        })

    return records


def _print_summary(records: list[dict]):
    sums: dict[tuple, list] = defaultdict(list)
    for r in records:
        if r["parse_ok"] and r["value"] is not None:
            sums[(r["seq_id"], r["outcome_id"], r["arm_id"])].append(r["value"])
    failures = sum(1 for r in records if not r["parse_ok"])
    print(f"\n── Arm means ({len(records)} records, {failures} parse failures) ──")
    for (seq_id, out_id, arm_id), vals in sorted(sums.items()):
        print(f"  seq={seq_id:>3} {out_id:<25} {arm_id:<40} "
              f"mean={sum(vals)/len(vals):.4f}  n={len(vals)}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int, default=50, dest="n_per_arm")
    parser.add_argument("--submit",   action="store_true",
                        help="Submit directly to OpenAI Batch API "
                             "(default: generate files for manual upload)")
    parser.add_argument("--async",    action="store_true", dest="run_async",
                        help="Run live via async API (immediate, full price)")
    parser.add_argument("--download", metavar="BATCH_ID", default=None)
    parser.add_argument("--list",     action="store_true")
    args = parser.parse_args()

    study_configs = load_study_configs(STUDIES_PATH)
    print(f"\nLoaded {len(study_configs)} study configs: {sorted(study_configs)}")

    if args.list:
        for seq_id, cfg in sorted(study_configs.items()):
            print(f"  seq={seq_id:>3}  arms={len(cfg['arms'])}  "
                  f"outcomes={len(cfg['outcomes'])}")
    elif args.download:
        download_batch(args.download, study_configs)
    elif args.run_async:
        asyncio.run(run_async(args.n_per_arm, study_configs))
    elif args.submit:
        submit_batch(args.n_per_arm, study_configs)
    else:
        generate_batch_file(args.n_per_arm, study_configs)
