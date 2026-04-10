# Data Directory Structure

All data flows through subdirectories following a clear progression from raw PDFs through analysis outputs.

## Directory Map

```
Data/
├── Papers/                      [Raw input]
│   ├── 12.pdf
│   ├── 19.pdf
│   └── ...
│   (symlink: ../US_Aggregate/Data/Papers/)
│
├── Ground_Truth/                [After step 01]
│   └── study_data.jsonl         One JSON line per study (design + GT effects)
│
├── Simulation/                  [After steps 02-03]
│   ├── Batch_Input/             Step 02 output (batch mode only)
│   │   ├── batch_no_reasoning_01_of_02.jsonl
│   │   ├── batch_no_reasoning_02_of_02.jsonl
│   │   └── ...
│   ├── Batch_Output/            Downloaded from OpenAI dashboard (batch mode only)
│   │   ├── *.jsonl
│   │   └── [unpack with step 03]
│   └── aggregate_simulation_raw_{cfg}.jsonl  [After step 02 async or step 03]
│       ├── aggregate_simulation_raw_no_reasoning.jsonl
│       ├── aggregate_simulation_raw_reasoning_low.jsonl
│       └── aggregate_simulation_raw_reasoning_medium.jsonl
│
├── Results/                     [After steps 04-05]
│   ├── effects_table_{cfg}.csv
│   │   ├── effects_table_no_reasoning.csv
│   │   ├── effects_table_reasoning_low.csv
│   │   └── effects_table_reasoning_medium.csv
│   └── effects_summary_{cfg}.txt
│       ├── effects_summary_no_reasoning.txt
│       ├── effects_summary_reasoning_low.txt
│       └── effects_summary_reasoning_medium.txt
│
└── Caches/                      [Intermediate caches]
    └── .extract_study_data_cache.json
        [Cached LLM responses from step 01]
```

---

## Papers/

**Status:** Raw input. Do not edit.

**Contents:**
- Study PDFs named `{seq_id}.pdf` or `{seq_id}_1.pdf`, `{seq_id}_2.pdf`, etc.
- Currently a symlink to `../US_Aggregate/Data/Papers/` (36 studies total)

**Who creates it:** Papers are sourced externally; stored in US_Aggregate for reuse across pipelines.

**Who reads it:** Script 01 (01_extract_study_data.py).

**Example filenames:**
```
12.pdf
19.pdf
30.pdf
47.pdf
# Some studies have multiple files:
103_1.pdf
103_2.pdf
```

---

## Ground_Truth/

**Status:** Output of step 01. Generated file.

**Contents:** `study_data.jsonl` — one JSON object per line, one per study.

### study_data.jsonl Schema

```json
{
  "seq_id": 103,
  "title": "Effects of Transparency on Trust in AI Agents",
  "paper_files": ["103.pdf"],
  "design_status": "ok|partial|failed",
  "results_status": "ok|partial|failed",
  "instrument": {
    "preamble": "You are a participant in a research study...",
    "control_arm_id": "control",
    "is_simulatable": true,
    "simulatability_note": "",
    "treatment_variations": [
      {
        "arm_id": "control",
        "arm_label": "Control group",
        "is_control": true,
        "text": "You are deciding whether to trust a recommendation..."
      },
      {
        "arm_id": "transparency_high",
        "arm_label": "High transparency",
        "is_control": false,
        "text": "You are deciding whether to trust a recommendation. The system shows: ..."
      }
    ],
    "outcome_questions": [
      {
        "outcome_id": "trust_score",
        "outcome_name": "Trust Score",
        "question_text": "How much do you trust this recommendation?",
        "response_instruction": "Reply with a single integer from 1 to 7 only.",
        "scale_type": "likert",
        "scale_min": 1,
        "scale_max": 7,
        "scale_labels": null
      }
    ]
  },
  "ground_truth": {
    "effects": [
      {
        "arm_id": "transparency_high",
        "outcome_id": "trust_score",
        "outcome_name": "Trust Score",
        "delta": 0.62,
        "treatment_mean": 5.82,
        "control_mean": 5.20,
        "n_treatment": 156,
        "n_control": 142,
        "metric": "mean",
        "note": "Table 3, Model 1"
      }
    ]
  }
}
```

### Key Fields

| Field | Type | Source | Usage |
|-------|------|--------|-------|
| `seq_id` | int | Paper metadata | Study identifier |
| `design_status` | str | Pass 1 extraction | "ok"/"partial"/"failed" |
| `results_status` | str | Pass 2 extraction | "ok"/"partial"/"failed"; "skipped" if no control |
| `instrument.control_arm_id` | str | Pass 1 extraction | Used by step 02 to identify control arm |
| `instrument.is_simulatable` | bool | Pass 1 extraction | Filters studies for simulation (step 02) |
| `ground_truth.effects[].arm_id` | str | Pass 1 + slugify | Matches arm_id from treatment_variations |
| `ground_truth.effects[].outcome_id` | str | Pass 1 + slugify | Matches outcome_id from outcome_questions |
| `ground_truth.effects[].delta` | float | Pass 2 extraction | Treatment effect = treatment_mean − control_mean |

### Shared ID Scheme

Both `treatment_variations` and `effects` use the same arm_ids and outcome_ids (slugified):
- "Control group" → `control_group`
- "Trust Score" → `trust_score`
- "Purchase Intent (1–7)" → `purchase_intent_1_7`

This shared scheme means downstream scripts join data via direct key lookup (no fuzzy matching).

### Quality Indicators

- `design_status == "ok"` → arm text, outcomes, scale info extracted cleanly
- `results_status == "ok"` → ≥ 75% of expected effects found (see Scripts/README.md)
- `results_status == "partial"` → 25–75% of expected effects found (usable, but gaps)
- `results_status == "failed"` → < 25% found; may need manual inspection

---

## Simulation/

**Status:** Output of step 02 (async mode) or step 02 + 03 (batch mode).

### Batch_Input/

**When used:** Batch API mode only (`02_simulate.py --generate-only`).

**Contents:** One or more `batch_{cfg}_NN_of_MM.jsonl` files.

**Format:** JSONL; each line is an OpenAI Batch API request:
```json
{
  "custom_id": "103__transparency_high__trust_score__0",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "gpt-4.1",
    "max_completion_tokens": 4096,
    "temperature": 1,
    "top_p": 1,
    "messages": [
      {"role": "system", "content": "You are a participant..."},
      {"role": "user", "content": "[preamble + arm text + question]"}
    ]
  }
}
```

**Workflow:**
1. Generated by step 02
2. Uploaded manually to OpenAI dashboard (or via `openai api files.create ...`)
3. Submitted as batch jobs
4. Downloaded from dashboard when complete → saved here
5. Unpacked by step 03 into aggregate_simulation_raw_{cfg}.jsonl

### Batch_Output/

**When used:** Batch API mode only (after download from OpenAI).

**Contents:** Downloaded batch result files (`*.jsonl`).

**Format:** JSONL; each line is an OpenAI Batch API result:
```json
{
  "custom_id": "103__transparency_high__trust_score__0",
  "response": {
    "status_code": 200,
    "request_id": "...",
    "body": {
      "id": "chatcmpl-...",
      "choices": [
        {
          "message": {
            "content": "5",
            "role": "assistant"
          }
        }
      ]
    }
  }
}
```

**Workflow:**
1. Downloaded from OpenAI dashboard
2. Placed here manually
3. Processed by step 03 (unpacking)

### aggregate_simulation_raw_{cfg}.jsonl

**Status:** Output of step 02 (async mode) or step 03 (batch mode unpacking).

**When created:**
- `aggregate_simulation_raw_no_reasoning.jsonl` → step 02 async with `--config no_reasoning`, or step 03 unpack
- `aggregate_simulation_raw_reasoning_low.jsonl` → step 02 async with `--config reasoning_low`, or step 03 unpack
- `aggregate_simulation_raw_reasoning_medium.jsonl` → step 02 async with `--config reasoning_medium`, or step 03 unpack

**Format:** JSONL; one line per simulated respondent:
```json
{
  "seq_id": 103,
  "arm_id": "transparency_high",
  "outcome_id": "trust_score",
  "pid": "103__transparency_high__trust_score__0",
  "response": "5",
  "value": 5.0,
  "parse_ok": true,
  "batch_cfg": "no_reasoning"
}
```

### Key Fields

| Field | Type | Meaning |
|-------|------|---------|
| `seq_id` | int | Study ID |
| `arm_id` | str | Arm ID (matches instrument.treatment_variations[].arm_id) |
| `outcome_id` | str | Outcome ID (matches instrument.outcome_questions[].outcome_id) |
| `response` | str | Raw LLM text response (e.g., "5", "YES") |
| `value` | float | Parsed response (e.g., 5.0, 1.0); None if parse failed |
| `parse_ok` | bool | Whether parsing succeeded |
| `batch_cfg` | str | Config used ("no_reasoning", "reasoning_low", "reasoning_medium") |
| `pid` | str | Participant ID (for tracking; usually custom_id from batch) |

**Usage:**
- Step 04 reads this file
- Groups by (seq_id, arm_id, outcome_id)
- Computes arm means: `mean([r["value"] for r in records if r["parse_ok"]])`

---

## Results/

**Status:** Output of step 04. Generated files.

### effects_table_{cfg}.csv

**Contents:** CSV with one row per study × treatment arm × outcome comparison.

**Columns:**
```
seq_id, study_label, arm_id, outcome_id, outcome_name, control_arm,
gt_delta, llm_effect,
gt_treatment_mean, gt_control_mean,
gt_n_treatment, gt_n_control,
metric, comparable, note
```

**Example row:**
```
103, "Effects of Transparency on Trust", transparency_high, trust_score, Trust Score, control,
0.62, 0.58,
5.82, 5.20,
156, 142,
mean, True, "Table 3"
```

**Key fields:**
- `gt_delta` — observed treatment effect (GT_Δ) from PDF
- `llm_effect` — predicted treatment effect (LLM_Δ) computed from sim arm means
- `comparable` — True if both control and treatment LLM data exist; False otherwise
- `note` — reason if not comparable (e.g., "llm control mean missing")

**Usage:**
- Used by step 05 for plotting
- Can be loaded into R, Python, Excel for further analysis
- Filter to `comparable==True` for correlation statistics

### effects_summary_{cfg}.txt

**Contents:** Human-readable summary statistics from step 04.

**Example output:**
```
Config       : no_reasoning
Contrasts (total)       : 145
Contrasts (comparable)  : 132
Studies with data       : 18

Pearson  r     = +0.427  (p=0.000)
Spearman r     = +0.389  (p=0.000)
RMSE           = 0.5834
Sign accuracy  = 74.1%   (81/109 non-zero contrasts)

Fisher z-transform weighted r = +0.431  (14 studies contributing)

Per-study breakdown:
  seq=103  Effects of Transparency on Trust           n= 8  r=+0.654  p=0.027  sign=100%
  seq=109  Information Framing Study                   n= 6  r=+0.340  p=0.480  sign=67%
  ...
```

---

## Caches/

**Status:** Intermediate storage; safe to delete (will be regenerated).

### .extract_study_data_cache.json

**Contents:** Cached responses from step 01 API calls.

**Format:** JSON object:
```json
{
  "p1__103__gpt-5.4": {
    "title": "...",
    "is_simulatable": true,
    "treatment_variations": [...],
    ...
  },
  "p2__103__gpt-5.4": [
    {
      "arm_id": "...",
      "outcome_id": "...",
      "delta": 0.62,
      ...
    }
  ]
}
```

**Keys:**
- `p1__{seq_id}__{model}` → Pass 1 (design) cache
- `p2__{seq_id}__{model}` → Pass 2 (results) cache

**Purpose:**
- Avoids re-calling OpenAI API for studies already extracted
- Safe to delete; step 01 will regenerate on next run (but will cost API credits)
- Use `--force` flag to bypass and re-extract anyway

---

## Data Flows

### Full Pipeline (All Steps)

```
Papers/ [raw PDFs]
  ↓ [step 01: extract_study_data.py]
Ground_Truth/study_data.jsonl
  ├── instrument (arms, outcomes) → [step 02]
  └── ground_truth.effects (GT Δ) → [step 04]

[step 02]
  ↓ (async mode)
Simulation/aggregate_simulation_raw_{cfg}.jsonl
  ↓
  (batch mode)
  ↓ Simulation/Batch_Input/*.jsonl
    [manual upload to OpenAI]
    Simulation/Batch_Output/*.jsonl (download)
      ↓ [step 03: unpack_batches.py]
      Simulation/aggregate_simulation_raw_{cfg}.jsonl

[step 04: compare_effects.py]
  Reads Ground_Truth/study_data.jsonl + Simulation/aggregate_simulation_raw_{cfg}.jsonl
  ↓
Results/effects_table_{cfg}.csv
Results/effects_summary_{cfg}.txt
  ↓ [step 05: plot.py]
  ../Figures/effects_{cfg}.pdf
```

---

## File Sizes (Typical)

| File | Size | Notes |
|------|------|-------|
| PDFs (all 36 studies) | ~500 MB | Varies; some >20 MB (scanned) |
| study_data.jsonl | ~2–3 MB | Design + GT effects for all studies |
| batch_*.jsonl input | ~100–200 MB per config | n=50 per arm; 7 files per config |
| aggregate_simulation_raw_*.jsonl | ~50–100 MB per config | ~7k–10k records per config |
| effects_table_*.csv | ~500 KB per config | One row per study × arm × outcome |
| effects_*.pdf | ~500 KB per config | Publication-quality figure |

---

## Cleanup & Maintenance

**Safe to delete (will regenerate if needed):**
- `Caches/.extract_study_data_cache.json` → deletes cached API responses; next run re-calls API
- `Simulation/Batch_Input/*` → if you've already uploaded and downloaded from OpenAI
- `Simulation/Batch_Output/*` → after unpacking with step 03

**Must preserve:**
- `Papers/*` — raw study PDFs
- `Ground_Truth/study_data.jsonl` — the extraction output; if deleted, must re-run step 01
- `Simulation/aggregate_simulation_raw_*.jsonl` — the simulation output; if deleted, must re-run step 02

**Archiving:**
For publication/sharing, include:
- `study_data.jsonl` (hand-extracted, minimal)
- `effects_table_{cfg}.csv` + `effects_summary_{cfg}.txt` (results summary)
- `../Figures/effects_{cfg}.pdf` (main figure)

Do NOT include:
- PDFs (copyright; reference via DOI)
- Batch input/output (intermediate; large)
- Simulation raw JSONL (intermediate; can regenerate if you have PDF + study_data.jsonl)

---

## Questions?

See [../README.md](../README.md) for project overview or [Scripts/README.md](../Scripts/README.md) for per-script details.
