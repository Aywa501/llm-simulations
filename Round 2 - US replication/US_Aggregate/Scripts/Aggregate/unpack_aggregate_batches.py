"""
unpack_aggregate_batches.py

Reads the downloaded batch output files from
    Data/Simulated_Data/Agregate_simulation_data/*.jsonl
and consolidates them into:
    Data/aggregate_simulation_raw_{config}.jsonl

This is the step that bridges the manual batch download (done because of
rate limits) and the compare/plot pipeline.  It applies the same parsers
that simulate_aggregate.py would have applied had the batches been
downloaded programmatically.

Usage:
    python unpack_aggregate_batches.py [--config no_reasoning] [--input-dir DIR]
"""

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

# Pull parsers + config loader from the sibling script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from simulate_aggregate import (
    load_study_configs, STUDIES_PATH, parse_integer
)

DATA_DIR     = SCRIPT_DIR.parents[1] / "Data"
DEFAULT_INDIR = DATA_DIR / "Simulation" / "Batch_Output"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--config", default="no_reasoning",
    help="Config label written into batch_cfg field (default: no_reasoning)",
)
parser.add_argument(
    "--input-dir", default=None,
    help="Directory containing batch *_output.jsonl files "
         f"(default: {DEFAULT_INDIR})",
)
args = parser.parse_args()

INPUT_DIR = Path(args.input_dir) if args.input_dir else DEFAULT_INDIR
OUT_PATH  = DATA_DIR / "Simulation" / f"aggregate_simulation_raw_{args.config}.jsonl"

# ---------------------------------------------------------------------------
# Build outcome lookup: (seq_id, arm_id, outcome_id) -> outcome config
# ---------------------------------------------------------------------------

study_configs = load_study_configs(STUDIES_PATH)
print(f"Loaded {len(study_configs)} study configs: {sorted(study_configs)}")

outcome_lookup: dict[tuple, dict] = {}
for seq_id, config in study_configs.items():
    for outcome in config["outcomes"]:
        for arm_id in config["arms"]:
            # Normalize: strip trailing underscores from slugs so they match
            # the values extracted from custom_ids (where __ splitting can
            # absorb trailing _ characters).
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

records   = []
api_errors = 0

for batch_file in batch_files:
    n_before = len(records)
    for line in open(batch_file):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)

        # Parse custom_id: seq_id__arm_id__outcome_id__i
        # arm_id can contain __ so we use positional parsing from both ends
        parts  = r["custom_id"].split("__")
        seq_id = int(parts[0])
        out_id = parts[-2]          # second-to-last
        # Strip any deduplication suffix so we perfectly mirror the lookup dictionary natively
        import re
        out_id = re.sub(r"_dup\d+$", "", out_id)
        # Strip leading/trailing underscores that arise from triple-underscore splits
        out_id = out_id.strip("_") or out_id
        arm_id = "__".join(parts[1:-2])  # everything between seq_id and out_id
        arm_id = arm_id.strip("_") or arm_id  # normalize to match lookup keys

        if r.get("error"):
            records.append({
                "seq_id":     seq_id,
                "arm_id":     arm_id,
                "outcome_id": out_id,
                "pid":        r["custom_id"],
                "response":   None,
                "value":      None,
                "parse_ok":   False,
                "batch_cfg":  args.config,
                "error":      str(r["error"]),
            })
            api_errors += 1
            continue

        text = (
            r["response"]["body"]["choices"][0]["message"]["content"] or ""
        ).strip()
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
            "batch_cfg":  args.config,
        })

    n_added = len(records) - n_before
    print(f"  {batch_file.name}: {n_added} records")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

parse_fails = sum(1 for r in records if not r["parse_ok"])
pct_fail    = 100 * parse_fails / len(records) if records else 0

print(f"\n── {len(records):,} records total ──")
print(f"   API errors    : {api_errors}")
print(f"   Parse failures: {parse_fails} ({pct_fail:.1f}%)")
print(f"   Parse successes: {len(records) - parse_fails - api_errors}")

# Arm-level means
sums: dict[tuple, list] = defaultdict(list)
for r in records:
    if r["parse_ok"] and r["value"] is not None:
        sums[(r["seq_id"], r["outcome_id"], r["arm_id"])].append(r["value"])

print("\n── Arm means (parsed records) ──")
for (seq_id, out_id, arm_id), vals in sorted(sums.items()):
    print(f"  seq={seq_id:>3}  {out_id:<45}  {arm_id:<40}  "
          f"mean={sum(vals)/len(vals):.4f}  n={len(vals)}")

# Studies not appearing in output (no records produced)
sim_seq_ids  = set(study_configs)
got_seq_ids  = {r["seq_id"] for r in records}
missing      = sim_seq_ids - got_seq_ids
if missing:
    print(f"\nWARNING: no records found for seq_ids: {sorted(missing)}")

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\nOutput → {OUT_PATH}  ({len(records):,} lines)")
