# US_Aggregate_2 — LLM Simulation Pipeline

Treatment-effects-based replication of Hewitt et al. "Predicting Results of Social Science Experiments Using Large Language Models" on a new behavioral economics dataset.

## Quick Start

```bash
cd Scripts

# 1. Extract study designs and ground-truth effects from PDFs
python 01_extract_study_data.py

# 2. Generate batch simulation input (or use async mode)
python 02_simulate.py --generate-only --all-batches --n 50

# 3. [MANUAL] Upload Batch_Input/*.jsonl files via OpenAI dashboard
#    Download completed batch outputs to Data/Simulation/Batch_Output/

# 4. Unpack downloaded batch results
python 03_unpack_batches.py --config no_reasoning

# 5. Compare LLM effects against ground truth
python 04_compare_effects.py --config no_reasoning

# 6. Generate effects figure
python 05_plot.py --config no_reasoning
```

## Overview

**Goal:** Predict whether LLMs can accurately simulate human behavior in behavioral economics experiments by comparing LLM-predicted treatment effects (Δ_LLM) against observed effects (Δ_GT) from published papers.

**Key Innovation:** This pipeline uses **treatment effects** (treatment_mean − control_mean), not raw arm means. This is the methodologically correct unit per Hewitt et al., and it naturally handles studies without single control arms.

**Architecture:**
- **Two-pass extraction** from PDFs: design (arms, outcomes, controls) then results (Δ values)
- **Unified data schema** (study_data.jsonl) with shared arm_ids and outcome_ids across instrument and ground_truth sections — no fuzzy matching needed
- **Treatment effects focus** throughout: compare predicted Δ against observed Δ
- **Flat script directory** with numbered stages (01–05) for clarity

## Pipeline Overview

```
PDFs (Data/Papers/)
   ↓
01_extract_study_data.py  [gpt-5.4 + medium reasoning]
   ↓
study_data.jsonl  (instrument + ground_truth effects with shared IDs)
   ↓
02_simulate.py  [gpt-4.1, no reasoning]
   ↓
aggregate_simulation_raw_{cfg}.jsonl  (per-respondent LLM responses)
   ↓
04_compare_effects.py
   ↓
effects_table_{cfg}.csv  (predicted Δ vs observed Δ)
   ↓
05_plot.py
   ↓
effects_{cfg}.pdf  (scatter + per-study r bars)
```

## Directory Structure

```
US_Aggregate_2/
├── Scripts/
│   ├── 01_extract_study_data.py    Design + GT effects from PDFs
│   ├── 02_simulate.py               LLM simulation (async or batch)
│   ├── 03_unpack_batches.py         Unpack batch output (optional)
│   ├── 04_compare_effects.py         Compare predicted vs observed Δ
│   ├── 05_plot.py                    Generate effects figure
│   └── README.md                     Detailed script guide
│
├── Data/
│   ├── Papers/                       → symlink to ../US_Aggregate/Data/Papers
│   ├── Ground_Truth/
│   │   └── study_data.jsonl         Output from 01_extract_study_data.py
│   ├── Simulation/
│   │   ├── Batch_Input/             Input files for manual batch submission
│   │   ├── Batch_Output/            Downloaded batch results
│   │   └── aggregate_simulation_raw_{cfg}.jsonl  Unpacked or direct async output
│   ├── Results/
│   │   ├── effects_table_{cfg}.csv   Output from 04_compare_effects.py
│   │   └── effects_summary_{cfg}.txt Summary statistics
│   └── Caches/
│       └── .extract_study_data_cache.json  Cached LLM extractions
│
├── Figures/
│   └── effects_{cfg}.pdf             Output from 05_plot.py
│
├── README.md                          This file
├── METHODOLOGY.md                     Pipeline design rationale
└── Data/README.md                     Data directory guide
```

## Key Concepts

### Treatment Effects vs Arm Means

**Old approach (arm means):**
- Correlate LLM_mean(arm) against human_mean(arm) for every arm
- Problem: conflates simulation quality with effect size; biased toward large-effect studies

**New approach (treatment effects):**
- Compute Δ = treatment_mean − control_mean for each treatment arm
- Correlate LLM_Δ against GT_Δ
- Matches Hewitt et al. methodology; properly measures causal prediction

### Shared IDs

Both `instrument` and `ground_truth.effects` sections of study_data.jsonl use the same:
- **arm_ids**: slugified from arm labels (e.g., "Control group" → "control_group")
- **outcome_ids**: slugified from outcome names (e.g., "Purchase Intent" → "purchase_intent")

This shared ID scheme means downstream scripts can join instrument and effects directly without fuzzy string matching.

### Models

- **Extraction (01):** `gpt-5.4` with `reasoning_effort="medium"` for precise design and effect identification
- **Simulation (02):** `gpt-4.1` with no reasoning (faster, cheaper, adequate for participant simulation)
- **Reasoning effort:** Only used for extraction; simulation uses standard temperature-based sampling

## Configuration

### Extraction

```bash
# Default: gpt-5.4 + medium reasoning
python 01_extract_study_data.py

# Re-extract (bypass cache)
python 01_extract_study_data.py --force

# Extract only design (skip results pass)
python 01_extract_study_data.py --pass1-only

# Specific studies only
python 01_extract_study_data.py --seq-ids 12 19 30
```

### Simulation

Three batch configurations (can run sequentially):

| Config | Model | Sampling | Use Case |
|--------|-------|----------|----------|
| `no_reasoning` | gpt-4.1 | temperature=1, top_p=1 | Baseline fast |
| `reasoning_low` | gpt-4.1 | reasoning_effort=low | Medium depth |
| `reasoning_medium` | gpt-4.1 | reasoning_effort=medium | Deep reasoning |

```bash
# Async mode (small runs, direct output)
python 02_simulate.py --mode async --config no_reasoning --n 50

# Batch mode (large runs, manual upload/download)
python 02_simulate.py --generate-only --all-batches --n 50
python 02_simulate.py --download BATCH_ID --config no_reasoning

# List loaded study configs
python 02_simulate.py --list
```

### Comparison & Plotting

```bash
# All studies
python 04_compare_effects.py --config no_reasoning

# Sound studies only (excludes seq 151, 164, 169, 174, 176)
python 04_compare_effects.py --config no_reasoning --sound-only

# Generate figure
python 05_plot.py --config no_reasoning --sound-only
```

## Output Files

### study_data.jsonl

One record per study with unified schema:
```json
{
  "seq_id": 103,
  "title": "...",
  "design_status": "ok",
  "results_status": "ok",
  "instrument": {
    "preamble": "...",
    "control_arm_id": "control",
    "is_simulatable": true,
    "treatment_variations": [
      {"arm_id": "treatment_a", "text": "...", "is_control": false}
    ],
    "outcome_questions": [
      {"outcome_id": "purchase_intent", "question_text": "...", "scale_min": 1, "scale_max": 7}
    ]
  },
  "ground_truth": {
    "effects": [
      {
        "arm_id": "treatment_a",
        "outcome_id": "purchase_intent",
        "delta": 0.45,
        "treatment_mean": 4.1,
        "control_mean": 3.65,
        "n_treatment": 150,
        "n_control": 148
      }
    ]
  }
}
```

### effects_table_{cfg}.csv

Comparison of predicted vs observed treatment effects:
- `seq_id`, `study_label`, `arm_id`, `outcome_id`
- `gt_delta`, `llm_effect`, `comparable` (boolean)
- `gt_treatment_mean`, `gt_control_mean`, metric metadata

### effects_summary_{cfg}.txt

Summary statistics:
- Pearson r, Spearman r, RMSE, sign accuracy
- Per-study breakdown with individual r values
- Fisher z-transform weighted r (accounts for study size)

### effects_{cfg}.pdf

Two-panel figure:
- **Top:** Scatter of LLM_Δ vs GT_Δ; colored by study; diagonal = perfect prediction; OLS fit line
- **Bottom:** Per-study Pearson r bars; annotated with Fisher z-weighted r

## Quality Checks

**Unsound studies** (excluded under `--sound-only`):
- **seq=151:** Visual matrix puzzles (LLM cannot see images)
- **seq=164:** Audio recording outcomes (LLM cannot hear)
- **seq=169:** Real-money dictator game (choice collapse, no signal)
- **seq=174:** Missing question text for all outcomes (blind answering)
- **seq=176:** Identical arm texts (no treatment variation)

## Troubleshooting

### "No PDF found"
Some studies have no PDF in Data/Papers/. Check papers are named correctly (seq_id only or seq_id_N).

### "Rate limits"
If extraction hits rate limits, the script auto-retries with exponential backoff (max 6 attempts). Results are cached, so you can safely re-run.

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY="sk-..."
python 01_extract_study_data.py
```

### Batch stuck at "processing"
Check the OpenAI dashboard. Batches can take 1+ hours. You can re-download anytime:
```bash
python 02_simulate.py --download BATCH_ID --config no_reasoning
```

### "Low coverage" warnings in extraction
If Pass 2 extraction finds < 25% of expected effects, the script retries automatically. If still low, the study may have ambiguous results or missing data tables.

## Design Rationale

See [METHODOLOGY.md](METHODOLOGY.md) for detailed discussion of:
- Why treatment effects instead of arm means
- Two-pass extraction design
- Shared ID schema
- Within-study normalization in earlier pipeline versions
- Fisher z-transform for meta-analytic r

## References

Hewitt, J. L., ... (2024). Predicting Results of Social Science Experiments Using Large Language Models. *Nature Human Behaviour*.

The original study compared raw arm means. This pipeline focuses on **treatment effects** to properly measure causal prediction accuracy — a more stringent and methodologically sound test.

---

**Questions?** See Scripts/README.md for per-script documentation.
