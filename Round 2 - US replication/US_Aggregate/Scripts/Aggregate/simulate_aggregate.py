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

import argparse, asyncio, io, json, random, re, time, uuid
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from openai import AsyncOpenAI, OpenAI, RateLimitError, APIConnectionError, APIError

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parents[1] / "Data"
MODEL      = "gpt-5.1"

# Point directly to the explicitly filtered whitelist
STUDIES_PATH = DATA_DIR / "Ground_Truth" / "simulatable_studies.jsonl"

# The boundary per batch input file chunk to safely stay under the queue token limit
MAX_TOKENS_PER_CHUNK = 1_345_000

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
    return DATA_DIR / "Simulation" / "Batch_Output" / f"aggregate_simulation_raw_{cfg_name}.jsonl"

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
    
    # 1. Custom Semantic Mappings (Specific failure captures)
    if "MESSAGE 1" in t: return 1.0
    if "MESSAGE 2" in t: return 0.0
    if "INCREASE" in t: return 1.0
    if "DECREASE" in t: return 0.0
    if "NEITHER" in t: return 0.5
    if "INFLATION" in t: return 1.0
    if "UNEMPLOYMENT" in t: return 0.0
    if "FOR" in t.split() or "FOR." in t: return 1.0
    if "AGAINST" in t.split() or "AGAINST." in t: return 0.0
    
    # Confidence / certainty labels (study 159 scale responses)
    if "VERY SURE" in t: return 1.0
    if "VERY UNSURE" in t: return 0.0
    if t == "SURE" or "SURE." in t: return 0.75
    if t == "UNSURE" or "UNSURE." in t: return 0.25
    
    # Engagement / extent labels (study 164 willingness_to_engage)
    if t == "ALL" or t.startswith("ALL "): return 1.0
    if t == "NONE" or t.startswith("NONE "): return 0.0
    if "SOME" in t.split() or "SOME." in t: return 0.5
    if "DON'T KNOW" in t or "DO NOT KNOW" in t: return 0.5
    
    # Reason views did not change (study 164)
    if "ALREADY KNEW" in t or "ALREADY CONSIDERED" in t: return 1.0
    if "DIDN'T FIND" in t or "NOT CONVINCING" in t or "UNCONVINCING" in t: return 0.0
    if t == "OTHER" or t.startswith("OTHER"): return 0.5
    
    # Macro variable names (study 178 free-text topic choice)
    if "INTEREST" in t: return 0.5
    if "PRICE" in t: return 1.0
    if "CONSUMPTION" in t: return 0.5
    if "EMPLOYMENT" in t or "LABOR" in t or "LABOUR" in t: return 0.0
    if "INCOME" in t or "WEALTH" in t: return 0.5
    if "STOCK" in t or "HOUSING" in t: return 0.5
    # Single-letter multiple-choice (C, D, E etc.)
    if t in ("C", "D", "E", "F", "G"): return 0.5
    
    # 2. Exact Affirmatives / Negatives
    if t.startswith("YES"): return 1.0
    if t.startswith("NO"):  return 0.0
    if re.search(r'\bYES\b', t): return 1.0
    if re.search(r'\bNO\b',  t): return 0.0
    
    # 3. Broader Policy Matrices & A/B Flags
    positive = r'\b(SUPPORT|FAVOR|AGREE|TRUE|ACCEPT|APPROVE|WILLING|1|A|OPTION A|JOB A)\b'
    negative = r'\b(OPPOSE|PREFER THE CURRENT|DISAGREE|FALSE|REJECT|DISAPPROVE|UNWILLING|0|B|OPTION B|JOB B)\b'
    
    if re.search(positive, t): return 1.0
    if re.search(negative, t): return 0.0
    
    # 4. Numeric fallback — if the text is just a bare number, return it
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
    # accept either 0-1 or 0-100 (normalise >1 values)
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
    # numeric fallback for responses like '5', '7'
    m = re.search(r'[-+]?\d+', text.strip())
    return float(int(m.group())) if m else None

def make_scale_parser(lo: float, hi: float):
    """Return a parser clamped to [lo, hi]. Falls back to verbal labels."""
    def parser(text: str) -> float | None:
        m = re.search(r"[-+]?\d+\.?\d*", text.strip())
        if m:
            return max(lo, min(hi, float(m.group())))
        # Verbal confidence / certainty fallback mapped to the scale
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
    # dollar handled via scale parser at load time
}

def resolve_parser(fmt: str, scale_min=None, scale_max=None):
    """Return the appropriate parser function for a given response_format."""
    if fmt in FIXED_PARSERS:
        return FIXED_PARSERS[fmt]
    if fmt in ("scale", "dollar") and scale_min is not None and scale_max is not None:
        return make_scale_parser(float(scale_min), float(scale_max))
    # For scale without bounds, or unknown format: try categorical then numeric
    def _flexible_parser(text: str) -> float | None:
        val = universal_categorical_parse(text)
        if val is not None:
            return val
        return parse_integer(text)
    return _flexible_parser

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

    studies = []
    with open(path) as f:
        for line in f:
            if line.strip():
                studies.append(json.loads(line))

    configs = {}

    for rec in studies:
        seq_id = rec["seq_id"]

        instrument = rec.get("instrument", {})
        if not instrument or not instrument.get("found"):
            print(f"  seq={seq_id}  skipped: no instrument found")
            continue

        variations = instrument.get("treatment_variations", [])
        if not variations:
            print(f"  seq={seq_id}  skipped: no treatment_variations")
            continue

        out_questions = instrument.get("outcome_questions", [])
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

        preamble = instrument.get("preamble") or ""

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

async def simulate_one(client, seq_id, arm_id, outcome, prompt, model_params,
                       retries: int = 6) -> dict:
    pid = str(uuid.uuid4())
    last_error: str = ""
    for attempt in range(retries):
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
        except RateLimitError as e:
            last_error = str(e)
            wait = min(120, 20 * 2 ** attempt) + random.uniform(0, 5)
            await asyncio.sleep(wait)
        except APIConnectionError as e:
            last_error = str(e)
            wait = min(30, 5 * 2 ** attempt) + random.uniform(0, 2)
            await asyncio.sleep(wait)
        except APIError as e:
            last_error = str(e)
            wait = min(60, 10 * 2 ** attempt) + random.uniform(0, 3)
            await asyncio.sleep(wait)
        except Exception as e:
            last_error = str(e)
            break   # non-API error (e.g. parse bug) — don't retry
    return {
        "seq_id":     seq_id,
        "arm_id":     arm_id,
        "outcome_id": outcome["id"],
        "pid":        pid,
        "response":   None,
        "value":      None,
        "parse_ok":   False,
        "error":      last_error,
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

def chunk_requests_by_tokens(requests: list[dict], max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[list[dict]]:
    """Split requests into chunks so each chunk stays safely under the queue token limit."""
    chunks = []
    current_chunk = []
    current_tokens = 0
    for r in requests:
        est_tokens = len(json.dumps(r)) // 4
        if current_tokens + est_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(r)
        current_tokens += est_tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def submit_batch(n_per_arm: int, cfg_name: str, study_configs: dict) -> list[str]:
    cfg          = BATCH_CONFIGS[cfg_name]
    model_params = cfg["model_params"]
    print(f"Initializing OpenAI client...")
    client       = OpenAI()
    print(f"Building request payloads...")
    requests     = build_batch_requests(n_per_arm, model_params, study_configs)
    
    chunks = chunk_requests_by_tokens(requests, max_tokens=MAX_TOKENS_PER_CHUNK)
    print(f"{cfg['label']}  |  {len(requests)} requests built, split into {len(chunks)} chunks")

    batch_ids = []
    for i, chunk in enumerate(chunks):
        content  = "\n".join(json.dumps(r) for r in chunk).encode()
        print(f"\n[Chunk {i+1}/{len(chunks)}] Uploading {len(content)/1024:.1f} KB JSONL to OpenAI Files API...")
        file_obj = client.files.create(file=io.BytesIO(content), purpose="batch")
        print(f"  File uploaded: id={file_obj.id}")

        print(f"  Submitting batch job...")
        batch    = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  Submitted  cfg={cfg_name}  batch_id={batch.id}  status={batch.status}")
        print(f"  Download:  python simulate_aggregate.py --download {batch.id} --config {cfg_name}")
        batch_ids.append(batch.id)
        time.sleep(1) # Small buffer between batch creations

    return batch_ids


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
        # Strip any appended duplicates so we map completely transparently
        import re
        out_id = re.sub(r"_dup\d+$", "", out_id)
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


def generate_batch_file(n_per_arm: int, cfg_name: str, study_configs: dict):
    """Write batch JSONL file(s) to disk dynamically chunked to safely stay under 3M queue token limit."""
    cfg          = BATCH_CONFIGS[cfg_name]
    model_params = cfg["model_params"]
    batch_dir    = DATA_DIR / "Simulation" / "Batch_Input"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear out old batch files for this config before generating new ones
    for old_file in batch_dir.glob(f"batch_{cfg_name}_*.jsonl"):
        old_file.unlink()

    requests = build_batch_requests(n_per_arm, model_params, study_configs)
    chunks   = chunk_requests_by_tokens(requests, max_tokens=MAX_TOKENS_PER_CHUNK)

    files_written = []

    for i, chunk in enumerate(chunks):
        out_file = batch_dir / f"batch_{cfg_name}_{i+1:02d}_of_{len(chunks):02d}.jsonl"
        with open(out_file, "w") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        size_kb = out_file.stat().st_size / 1024
        est_tokens = sum(len(json.dumps(r)) for r in chunk) // 4
        print(f"  File {i+1}/{len(chunks)}: {len(chunk):>5} requests  "
              f"{size_kb:>7.1f} KB  ~{est_tokens:>7,} tokens  → {out_file.name}")
        files_written.append(out_file)

    print(f"\n{cfg['label']}  |  n={n_per_arm}  |  {len(chunks)} files dynamically chunked")
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
    parser.add_argument("--mode",        choices=["async", "batch"], default="batch")
    parser.add_argument("--config",      choices=list(BATCH_CONFIGS), default="no_reasoning")
    parser.add_argument("--all-batches", action="store_true",
                        help="Submit all 3 batch configs in sequence")
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate batch JSONL file(s) without uploading to API")
    parser.add_argument("--download",    metavar="BATCH_ID", default=None)
    parser.add_argument("--list",        action="store_true",
                        help="List loaded study configs and exit")
    parser.add_argument("--studies-file", metavar="FILE", default=None,
                        help="Override default simulatable_studies.json path "
                             "(e.g. simulatable_studies_v2.json)")
    args = parser.parse_args()

    studies_path = (DATA_DIR / args.studies_file
                    if args.studies_file else STUDIES_PATH)
    study_configs = load_study_configs(studies_path)
    print(f"\nLoaded {len(study_configs)} study configs from {studies_path.name}: "
          f"{sorted(study_configs)}")

    if args.list:
        for seq_id, cfg in sorted(study_configs.items()):
            n_arms = len(cfg["arms"])
            n_out  = len(cfg["outcomes"])
            print(f"  seq={seq_id:>3}  arms={n_arms}  outcomes={n_out}")
    elif args.download:
        download_batch(args.download, args.config, study_configs)
    elif args.generate_only:
        if args.all_batches:
            for cfg_name in BATCH_CONFIGS:
                generate_batch_file(args.n_per_arm, cfg_name, study_configs)
        else:
            generate_batch_file(args.n_per_arm, args.config, study_configs)
    elif args.all_batches:
        for cfg_name in BATCH_CONFIGS:
            submit_batch(args.n_per_arm, cfg_name, study_configs)
    elif args.mode == "batch":
        submit_batch(args.n_per_arm, args.config, study_configs)
    else:
        asyncio.run(run_async(args.n_per_arm, args.config, study_configs))


