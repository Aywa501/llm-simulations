"""
03_unpack_batches.py  —  Step 3 of the pipeline (batch API only)

Reads manually-downloaded batch output files from
    Data/Simulation/Batch_Output/*.jsonl

and consolidates them into:
    Data/Simulation/aggregate_simulation_raw_{config}.jsonl

Skip this step if you used `02_simulate.py --mode async` or
`02_simulate.py --download BATCH_ID`, which write the output file directly.

Usage:
    python 03_unpack_batches.py [--config no_reasoning] [--input-dir DIR]
"""

import argparse, importlib.util, json, re, sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR    = Path(__file__).resolve().parent
DATA_DIR      = SCRIPT_DIR.parent / "Data"
DEFAULT_INDIR = DATA_DIR / "Simulation" / "Batch_Output"

# ---------------------------------------------------------------------------
# Import helpers from 02_simulate.py (load_study_configs, parse_integer)
# ---------------------------------------------------------------------------

_spec = importlib.util.spec_from_file_location(
    "simulate", SCRIPT_DIR / "02_simulate.py"
)
_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sim)

load_study_configs = _sim.load_study_configs
parse_integer      = _sim.parse_integer
STUDIES_PATH       = _sim.STUDIES_PATH

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", default=None,
                    help=f"Directory with batch output JSONL files "
                         f"(default: {DEFAULT_INDIR})")
args = parser.parse_args()

INPUT_DIR = Path(args.input_dir) if args.input_dir else DEFAULT_INDIR
OUT_PATH  = DATA_DIR / "Simulation" / "aggregate_simulation_raw.jsonl"

# ---------------------------------------------------------------------------
# Build outcome lookup: (seq_id, arm_id, outcome_id) → outcome config
# ---------------------------------------------------------------------------

study_configs = load_study_configs(STUDIES_PATH)
print(f"Loaded {len(study_configs)} study configs: {sorted(study_configs)}")

outcome_lookup: dict[tuple, dict] = {}
for seq_id, config in study_configs.items():
    for outcome in config["outcomes"]:
        for arm_id in config["arms"]:
            norm_arm = arm_id.strip("_")
            norm_out = outcome["id"].strip("_")
            outcome_lookup[(seq_id, norm_arm, norm_out)] = outcome

# ---------------------------------------------------------------------------
# Find batch output files
# ---------------------------------------------------------------------------

batch_files = sorted(INPUT_DIR.glob("*.jsonl"))
if not batch_files:
    print(f"No *.jsonl files found in {INPUT_DIR}")
    sys.exit(1)

print(f"\nFound {len(batch_files)} batch output file(s) in {INPUT_DIR.name}/")

# ---------------------------------------------------------------------------
# Unpack
# ---------------------------------------------------------------------------

records:    list[dict] = []
api_errors: int        = 0

for batch_file in batch_files:
    n_before = len(records)
    for line in open(batch_file):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)

        # custom_id format: seq_id__arm_id__outcome_id__i
        parts  = r["custom_id"].split("__")
        seq_id = int(parts[0])
        out_id = re.sub(r"_dup\d+$", "", parts[-2]).strip("_")
        arm_id = "__".join(parts[1:-2]).strip("_")

        if r.get("error"):
            records.append({
                "seq_id":     seq_id,
                "arm_id":     arm_id,
                "outcome_id": out_id,
                "pid":        r["custom_id"],
                "response":   None,
                "value":      None,
                "parse_ok":   False,
                "error":      str(r["error"]),
            })
            api_errors += 1
            continue

        text    = (r["response"]["body"]["choices"][0]["message"]["content"]
                   or "").strip()
        outcome = outcome_lookup.get((seq_id, arm_id, out_id))
        parser  = outcome["_parser"] if outcome else parse_integer
        value   = parser(text)

        records.append({
            "seq_id":     seq_id,
            "arm_id":     arm_id,
            "outcome_id": out_id,
            "pid":        r["custom_id"],
            "response":   text,
            "value":      value,
            "parse_ok":   value is not None,
        })

    print(f"  {batch_file.name}: {len(records) - n_before} records")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

parse_fails = sum(1 for r in records if not r["parse_ok"])
pct_fail    = 100 * parse_fails / len(records) if records else 0

print(f"\n── {len(records):,} records total ──")
print(f"   API errors     : {api_errors}")
print(f"   Parse failures : {parse_fails} ({pct_fail:.1f}%)")
print(f"   Parse successes: {len(records) - parse_fails - api_errors}")

sums: dict[tuple, list] = defaultdict(list)
for r in records:
    if r["parse_ok"] and r["value"] is not None:
        sums[(r["seq_id"], r["outcome_id"], r["arm_id"])].append(r["value"])

print("\n── Arm means (parsed records) ──")
for (seq_id, out_id, arm_id), vals in sorted(sums.items()):
    print(f"  seq={seq_id:>3}  {out_id:<40}  {arm_id:<35}  "
          f"mean={sum(vals)/len(vals):.4f}  n={len(vals)}")

missing = set(study_configs) - {r["seq_id"] for r in records}
if missing:
    print(f"\nWARNING: no records for seq_ids: {sorted(missing)}")

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\nOutput → {OUT_PATH}  ({len(records):,} lines)")
