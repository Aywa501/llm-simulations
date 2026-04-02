"""
simulate_aggregate.py

Participant-mode LLM simulation for all simulatable aggregate studies.

Reads study configs from Data/simulatable_studies.json
(produced by the simulatability-filtering step). Each entry contains
preamble, treatment_variations, outcome_questions, and ground-truth results.

Outputs per batch config:
    Data/aggregate_simulation_raw_{cfg}.jsonl

Usage:
    python simulate_aggregate.py --mode async  --config no_reasoning [--n 50]
    python simulate_aggregate.py --mode batch  --config no_reasoning [--n 50]
    python simulate_aggregate.py --all-batches [--n 50]
    python simulate_aggregate.py --download BATCH_ID --config no_reasoning
    python simulate_aggregate.py --list          # show loaded study configs
"""

import argparse, asyncio, io, json, re, time, uuid
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from openai import AsyncOpenAI, OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parents[1] / "Data"
MODEL      = "gpt-5.1"

STUDIES_PATH = DATA_DIR / "simulatable_studies.json"

# ---------------------------------------------------------------------------
# Batch configurations  (identical to microdata pipeline)
# ---------------------------------------------------------------------------

BATCH_CONFIGS = {
    "no_reasoning": {
        "label":        "Batch 1 — no reasoning",
        "model_params": {"temperature": 1, "top_p": 1},
    },
    "reasoning_low": {
        "label":        "Batch 2 — reasoning low",
        "model_params": {"reasoning_effort": "low"},
    },
    "reasoning_medium": {
        "label":        "Batch 3 — reasoning medium",
        "model_params": {"reasoning_effort": "medium"},
    },
}

def output_path(cfg_name: str) -> Path:
    return DATA_DIR / f"aggregate_simulation_raw_{cfg_name}.jsonl"

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

def parse_binary(text: str) -> float | None:
    t = text.strip().upper()
    if t.startswith("YES"): return 1.0
    if t.startswith("NO"):  return 0.0
    if re.search(r'\bYES\b', t): return 1.0
    if re.search(r'\bNO\b',  t): return 0.0
    return None

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
    # accept either 0-1 or 0-100 (normalise >1 values)
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))

def parse_integer(text: str) -> float | None:
    m = re.search(r"[-+]?\d+", text.strip())
    return float(int(m.group())) if m else None

def parse_choice_ab(text: str) -> float | None:
    t = text.strip().upper()
    if t.startswith("A") or "JOB A" in t or "OPTION A" in t: return 1.0
    if t.startswith("B") or "JOB B" in t or "OPTION B" in t: return 0.0
    if re.search(r'\bA\b', t): return 1.0
    if re.search(r'\bB\b', t): return 0.0
    return None

def make_scale_parser(lo: float, hi: float):
    """Return a parser clamped to [lo, hi]."""
    def parser(text: str) -> float | None:
        m = re.search(r"[-+]?\d+\.?\d*", text.strip())
        if not m:
            return None
        return max(lo, min(hi, float(m.group())))
    return parser

FIXED_PARSERS = {
    "binary":     parse_binary,
    "percent":    parse_percent,
    "proportion": parse_proportion,
    "integer":    parse_integer,
    "choice":     parse_choice_ab,
    "choice_ab":  parse_choice_ab,
    # dollar handled via scale parser at load time
}

def resolve_parser(fmt: str, scale_min=None, scale_max=None):
    """Return the appropriate parser function for a given response_format."""
    if fmt in FIXED_PARSERS:
        return FIXED_PARSERS[fmt]
    if fmt in ("scale", "dollar") and scale_min is not None and scale_max is not None:
        return make_scale_parser(float(scale_min), float(scale_max))
    # generic numeric fallback
    return parse_integer

# ---------------------------------------------------------------------------
# Load study configs from simulatable_studies.json
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50]

def load_study_configs(path: Path) -> dict[int, dict]:
    """
    Parse simulatable_studies.json into STUDY_CONFIGS dict
    keyed by seq_id, matching the structure expected by the simulation runner.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run the simulatability filtering step first."
        )

    with open(path) as f:
        studies = json.load(f)

    configs = {}

    for rec in studies:
        seq_id = rec["seq_id"]

        variations = rec.get("treatment_variations", [])
        if not variations:
            print(f"  seq={seq_id}  skipped: no treatment_variations")
            continue

        out_questions = rec.get("outcome_questions", [])
        if not out_questions:
            print(f"  seq={seq_id}  skipped: no outcome_questions")
            continue

        # Build arms dict: arm_id → arm text
        arms: dict[str, str] = {}
        for v in variations:
            arm_id = v.get("arm_id") or slugify(v.get("arm_label", "arm"))
            arms[arm_id] = v.get("text", "")

        # Build outcomes list
        outcomes = []
        for q in out_questions:
            fmt = q.get("response_format", "integer")
            lo  = q.get("scale_min")
            hi  = q.get("scale_max")
            outcomes.append({
                "id":              slugify(q.get("outcome_name", "outcome")),
                "response_format": fmt,
                "question":        "\n\n" + q.get("response_instruction", "Reply with a number."),
                "_parser":         resolve_parser(fmt, lo, hi),
            })

        preamble = rec.get("preamble") or ""

        configs[seq_id] = {
            "seq_id":   seq_id,
            "preamble": preamble,
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
# Simulation runner  (identical pattern to microdata)
# ---------------------------------------------------------------------------

async def simulate_one(client, seq_id, arm_id, outcome, prompt, model_params) -> dict:
    pid = str(uuid.uuid4())
    try:
        r = await client.chat.completions.create(
            model=MODEL,
            **model_params,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        text   = (r.choices[0].message.content or "").strip()
        parser = outcome["_parser"]
        value  = parser(text)
        return {
            "seq_id":     seq_id,
            "arm_id":     arm_id,
            "outcome_id": outcome["id"],
            "pid":        pid,
            "response":   text,
            "value":      value,
            "parse_ok":   value is not None,
        }
    except Exception as e:
        return {
            "seq_id":     seq_id,
            "arm_id":     arm_id,
            "outcome_id": outcome["id"],
            "pid":        pid,
            "response":   None,
            "value":      None,
            "parse_ok":   False,
            "error":      str(e),
        }


async def run_study(client, config, n_per_arm, writer, pbar, model_params):
    seq_id   = config["seq_id"]
    outcomes = config["outcomes"]
    tasks    = []
    for arm_id in config["arms"]:
        for outcome in outcomes:
            prompt = build_prompt(config, arm_id, outcome)
            for _ in range(n_per_arm):
                tasks.append(
                    simulate_one(client, seq_id, arm_id, outcome, prompt, model_params)
                )
    for coro in asyncio.as_completed(tasks):
        rec = await coro
        writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pbar.update(1)


def print_summary(records: list[dict]):
    sums = defaultdict(list)
    for r in records:
        if r["parse_ok"] and r["value"] is not None:
            sums[(r["seq_id"], r["outcome_id"], r["arm_id"])].append(r["value"])
    failures = sum(1 for r in records if not r["parse_ok"])
    print(f"\n── Simulated arm means ({len(records)} records, {failures} parse failures) ──")
    for (seq_id, out_id, arm_id), vals in sorted(sums.items()):
        print(f"  seq={seq_id:>3} {out_id:<20} {arm_id:<45} "
              f"mean={sum(vals)/len(vals):.4f}  n={len(vals)}")

# ---------------------------------------------------------------------------
# Async mode
# ---------------------------------------------------------------------------

async def run_async(n_per_arm: int, cfg_name: str, study_configs: dict):
    cfg          = BATCH_CONFIGS[cfg_name]
    model_params = cfg["model_params"]
    out          = output_path(cfg_name)
    client       = AsyncOpenAI()
    total        = sum(len(c["arms"]) * len(c["outcomes"]) * n_per_arm
                       for c in study_configs.values())
    seq_ids = list(study_configs)
    print(f"{cfg['label']}  |  studies={seq_ids}  |  n={n_per_arm}  |  calls={total}")
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        with tqdm(total=total, unit="call") as pbar:
            for seq_id, config in study_configs.items():
                pbar.set_description(f"seq={seq_id}")
                await run_study(client, config, n_per_arm, f, pbar, model_params)

    with open(out) as f:
        records = [json.loads(l) for l in f]
    print_summary(records)
    print(f"Output → {out}")

# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def build_batch_requests(n_per_arm: int, model_params: dict,
                         study_configs: dict) -> list[dict]:
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
                            **model_params,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user",   "content": prompt},
                            ],
                        },
                    })
    return requests


def submit_batch(n_per_arm: int, cfg_name: str, study_configs: dict) -> str:
    cfg          = BATCH_CONFIGS[cfg_name]
    model_params = cfg["model_params"]
    print(f"Initializing OpenAI client...")
    client       = OpenAI()
    print(f"Building request payloads...")
    requests     = build_batch_requests(n_per_arm, model_params, study_configs)
    print(f"{cfg['label']}  |  {len(requests)} requests built")

    content  = "\n".join(json.dumps(r) for r in requests).encode()
    print(f"Uploading {len(content)/1024:.1f} KB JSONL to OpenAI Files API...")
    file_obj = client.files.create(file=io.BytesIO(content), purpose="batch")
    print(f"File uploaded: id={file_obj.id}")

    print(f"Submitting batch job...")
    batch    = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Submitted  cfg={cfg_name}  batch_id={batch.id}  status={batch.status}")
    print(f"Download:  python simulate_aggregate.py --download {batch.id} --config {cfg_name}")
    return batch.id


def download_batch(batch_id: str, cfg_name: str, study_configs: dict):
    client = OpenAI()
    out    = output_path(cfg_name)
    print(f"Checking batch {batch_id}  cfg={cfg_name} …")
    while True:
        batch  = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  status={batch.status}  completed={counts.completed}/{counts.total}  "
              f"failed={counts.failed}")
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status: {batch.status}")
        time.sleep(30)

    if batch.output_file_id is None:
        if batch.error_file_id:
            err_raw = client.files.content(batch.error_file_id).text
            for s in err_raw.splitlines()[:3]:
                print("  ERROR SAMPLE:", s[:300])
        raise RuntimeError(f"Batch {batch_id}: 0 successes. See error samples above.")

    raw     = client.files.content(batch.output_file_id).text
    records = []

    # Build lookup: (seq_id, arm_id, outcome_id) → outcome config for parser
    outcome_lookup: dict[tuple, dict] = {}
    for seq_id, config in study_configs.items():
        for outcome in config["outcomes"]:
            for arm_id in config["arms"]:
                outcome_lookup[(seq_id, arm_id, outcome["id"])] = outcome

    for line in raw.splitlines():
        if not line.strip():
            continue
        r      = json.loads(line)
        parts  = r["custom_id"].split("__")
        seq_id = int(parts[0])
        out_id = parts[-2]
        arm_id = "__".join(parts[1:-2])

        if r.get("error"):
            records.append({"seq_id": seq_id, "arm_id": arm_id, "outcome_id": out_id,
                             "pid": r["custom_id"], "response": None,
                             "value": None, "parse_ok": False, "error": str(r["error"])})
        else:
            text    = (r["response"]["body"]["choices"][0]["message"]["content"] or "").strip()
            outcome = outcome_lookup.get((seq_id, arm_id, out_id))
            parser  = outcome["_parser"] if outcome else parse_integer
            value   = parser(text)
            records.append({"seq_id": seq_id, "arm_id": arm_id, "outcome_id": out_id,
                             "pid": r["custom_id"], "response": text,
                             "value": value, "parse_ok": value is not None,
                             "batch_cfg": cfg_name})

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print_summary(records)
    print(f"Output → {out}")


def generate_batch_file(n_per_arm: int, per_file: int, cfg_name: str, study_configs: dict):
    """Write batch JSONL file(s) to disk without calling the API.
    
    Splits n_per_arm into chunks of per_file to stay under token limits.
    e.g. --n 50 --per-file 4 → 13 batch files (12×4 + 1×2).
    """
    cfg          = BATCH_CONFIGS[cfg_name]
    model_params = cfg["model_params"]
    batch_dir    = DATA_DIR / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Figure out how many files we need
    n_files = (n_per_arm + per_file - 1) // per_file
    files_written = []

    for i in range(n_files):
        chunk_n = min(per_file, n_per_arm - i * per_file)
        requests = build_batch_requests(chunk_n, model_params, study_configs)
        out_file = batch_dir / f"batch_{cfg_name}_{i+1:02d}_of_{n_files:02d}.jsonl"
        with open(out_file, "w") as f:
            for r in requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        size_kb = out_file.stat().st_size / 1024
        est_tokens = sum(len(json.dumps(r)) for r in requests) // 4
        print(f"  File {i+1}/{n_files}: {len(requests):>5} requests  "
              f"{size_kb:>7.1f} KB  ~{est_tokens:>7,} tokens  (n={chunk_n})  → {out_file.name}")
        files_written.append(out_file)

    print(f"\n{cfg['label']}  |  n={n_per_arm}  |  {n_files} files  |  {per_file}/file")
    print(f"Output dir: {batch_dir}")
    print(f"\nUpload each file manually via OpenAI dashboard or:")
    for f in files_written:
        print(f"  openai api files.create -f {f} -p batch")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=50, dest="n_per_arm",
                        help="Total simulated subjects per arm (default: 50)")
    parser.add_argument("--per-file",    type=int, default=4,
                        help="Subjects per arm per batch file (default: 4, fits ~900k tokens)")
    parser.add_argument("--mode",        choices=["async", "batch"], default="batch")
    parser.add_argument("--config",      choices=list(BATCH_CONFIGS), default="no_reasoning")
    parser.add_argument("--all-batches", action="store_true",
                        help="Submit all 3 batch configs in sequence")
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate batch JSONL file(s) without uploading to API")
    parser.add_argument("--download",    metavar="BATCH_ID", default=None)
    parser.add_argument("--list",        action="store_true",
                        help="List loaded study configs and exit")
    args = parser.parse_args()

    study_configs = load_study_configs(STUDIES_PATH)
    print(f"\nLoaded {len(study_configs)} study configs: {sorted(study_configs)}")

    if args.list:
        for seq_id, cfg in sorted(study_configs.items()):
            n_arms = len(cfg["arms"])
            n_out  = len(cfg["outcomes"])
            print(f"  seq={seq_id:>3}  arms={n_arms}  outcomes={n_out}")
    elif args.generate_only:
        generate_batch_file(args.n_per_arm, args.per_file, args.config, study_configs)
    elif args.download:
        download_batch(args.download, args.config, study_configs)
    elif args.all_batches:
        for cfg_name in BATCH_CONFIGS:
            submit_batch(args.n_per_arm, cfg_name, study_configs)
    elif args.mode == "batch":
        submit_batch(args.n_per_arm, args.config, study_configs)
    else:
        asyncio.run(run_async(args.n_per_arm, args.config, study_configs))


