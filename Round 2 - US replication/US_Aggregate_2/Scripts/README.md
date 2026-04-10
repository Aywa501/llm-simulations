# Scripts — Detailed Documentation

Five numbered scripts form the complete pipeline from PDF extraction to visualization.

## 01_extract_study_data.py

**Purpose:** Extract experimental design and ground-truth treatment effects from study PDFs using LLM + two-pass approach.

**Models:** `gpt-5.4` with `reasoning_effort="medium"`

**Input:**
- Study PDFs from `Data/Papers/*.pdf`
- Named: `{seq_id}.pdf` or `{seq_id}_1.pdf`, `{seq_id}_2.pdf`, etc.

**Output:**
- `Data/Ground_Truth/study_data.jsonl` — one JSON line per study
- `Data/Caches/.extract_study_data_cache.json` — cached API responses (safe to re-run)

### Two-Pass Extraction

**Pass 1 — Design:**
LLM reads the paper and returns:
- `title` — full paper title
- `control_arm_label` — which condition is the control/baseline
- `treatment_variations[]` — all experimental arms with exact stimulus text
- `outcome_questions[]` — primary dependent variables (question text, scale type, min/max)
- `is_simulatable` — boolean: can an LLM answer these questions from text alone?
- `preamble` — shared introduction shown to all participants

**Pass 2 — Results:**
Using arm_ids and outcome_ids from Pass 1, LLM extracts:
- `delta` — treatment_mean − control_mean for each treatment arm × outcome
- `treatment_mean`, `control_mean` — raw values
- `n_treatment`, `n_control` — sample sizes
- `metric` — "mean", "proportion", "coefficient", or "other"
- `note` — source table or caveat

Both passes return JSON only (no prose). Coverage < 25% triggers automatic retry.

### Schema

```json
{
  "seq_id": 103,
  "title": "...",
  "paper_files": ["103.pdf"],
  "design_status": "ok|partial|failed",
  "results_status": "ok|partial|failed",
  "instrument": {
    "preamble": "...",
    "control_arm_id": "control",
    "is_simulatable": true,
    "simulatability_note": "",
    "treatment_variations": [
      {
        "arm_id": "treatment_a",
        "arm_label": "Treatment A",
        "is_control": false,
        "text": "stimulus text here"
      }
    ],
    "outcome_questions": [
      {
        "outcome_id": "purchase_intent",
        "outcome_name": "Purchase Intent",
        "question_text": "How likely are you to purchase?",
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
        "arm_id": "treatment_a",
        "outcome_id": "purchase_intent",
        "outcome_name": "Purchase Intent",
        "delta": 0.45,
        "treatment_mean": 4.1,
        "control_mean": 3.65,
        "n_treatment": 150,
        "n_control": 148,
        "metric": "mean",
        "note": "Table 2, col 3"
      }
    ]
  }
}
```

### Usage

```bash
# Extract all studies (uses cache for previous successes)
python 01_extract_study_data.py

# Force re-extract everything (bypass cache)
python 01_extract_study_data.py --force

# Design pass only (skip results extraction)
python 01_extract_study_data.py --pass1-only

# Specific studies
python 01_extract_study_data.py --seq-ids 12 19 30 47

# Change model (default: gpt-5.4)
python 01_extract_study_data.py --model gpt-4.1
```

### Performance

- ~90 seconds per study for full extraction (Pass 1 + Pass 2) with gpt-5.4 + medium reasoning
- Rate limit handling: auto-retry up to 6 times with exponential backoff (20–120s per retry)
- Concurrency limit: 4 concurrent API calls to stay within rate limits
- Cache prevents re-extraction — safe to re-run; only new/forced studies are processed

### Key Implementation Details

- **PDF text extraction:** pdfplumber; max 90k chars per study (truncated if needed)
- **JSON parsing:** Handles markdown code blocks and incomplete JSON objects
- **Coverage grading:**
  - ≥ 75% of expected effects: status="ok"
  - 25–75%: status="partial" (study still usable, but incomplete)
  - < 25%: status="failed" (triggers automatic retry)
- **Slugify function:** Shared across all scripts; converts text to lowercase, alphanumeric + underscores, max 50 chars

---

## 02_simulate.py

**Purpose:** Generate LLM participant responses for each arm × outcome combination.

**Models:** `gpt-4.1` (no reasoning; temperature/reasoning_effort config varies by batch)

**Input:**
- `Data/Ground_Truth/study_data.jsonl` (output from script 01)

**Output:**
- **Async mode:** `Data/Simulation/aggregate_simulation_raw_{cfg}.jsonl` (direct)
- **Batch mode:**
  - `Data/Simulation/Batch_Input/*.jsonl` (upload manually)
  - `Data/Simulation/Batch_Output/*.jsonl` (download from dashboard; unpack with script 03)

### Three Batch Configurations

| Config | Temperature | Reasoning | Use |
|--------|-------------|-----------|-----|
| `no_reasoning` | 1.0, top_p=1 | none | Baseline, fast |
| `reasoning_low` | — | low | Medium depth |
| `reasoning_medium` | — | medium | Deep reasoning (slower) |

Each can be run independently. Results are separate output files. Compare all three to assess impact of reasoning effort.

### Running Modes

**Async (small runs, slower API but direct download):**
```bash
python 02_simulate.py --mode async --config no_reasoning --n 50
```
Directly produces `aggregate_simulation_raw_no_reasoning.jsonl`.

**Batch API (large runs, manual upload/download):**
```bash
# Generate batch input files
python 02_simulate.py --generate-only --config no_reasoning --n 50

# Then manually upload Batch_Input/*.jsonl to OpenAI dashboard
# Download .jsonl from dashboard once complete
# Save to Data/Simulation/Batch_Output/

# Unpack into raw simulation output
python 03_unpack_batches.py --config no_reasoning
```

**Batch API (all configs):**
```bash
python 02_simulate.py --generate-only --all-batches --n 50
```
Generates input files for no_reasoning, reasoning_low, reasoning_medium.

**Check batch status:**
```bash
python 02_simulate.py --download BATCH_ID --config no_reasoning
```
Polls OpenAI API until completion, then downloads and unpacks.

**List loaded studies:**
```bash
python 02_simulate.py --list
```
Shows seq_id, # arms, # outcomes for each simulatable study.

### Participant Simulation

For each arm × outcome pair:

1. **Prompt:** preamble + arm_text + outcome question
2. **System message:** "You are a participant in an online survey. Give only exact response format requested."
3. **LLM response:** Text (e.g., "5", "YES", "purchase")
4. **Parse:** Response-format-specific parser converts text → float
5. **Record:** {seq_id, arm_id, outcome_id, value, parse_ok}

### Response Parsers

Specialized parsers for each response format (defined in resolve_parser):

- **binary:** "YES"/"NO" → 1.0/0.0; also handles "SUPPORT"/"OPPOSE", etc.
- **percent:** Extracts number and clamps to [0, 100]
- **proportion:** Extracts number, normalizes to [0, 1]
- **integer:** Extracts integer
- **likert/continuous:** Scale parser — clamps to [min, max], falls back to verbal labels
- **categorical:** Multi-choice; maps options to positions
- **choice_ab:** Binary choice (A/B, Option A/B, Job A/B)
- **other:** universal_categorical_parse — broad pattern matching

See `universal_categorical_parse()` for specific mappings (e.g., "INFLATION" → 1.0, "UNEMPLOYMENT" → 0.0).

### Batch Request Format

Custom_id format: `{seq_id}__{arm_id}__{outcome_id}__{i}`

Example: `103__treatment_a__purchase_intent__0`

Batch API automatically handles chunking to stay under 3M queue token limit.

### Performance

- **Async:** ~10–30 seconds per study (varies with # arms × outcomes)
- **Batch API:** No rate limits; typically completes in 1–2 hours for large batches
- Parse failure rate: typically 1–5% (grammatical errors, non-standard response format)

---

## 03_unpack_batches.py

**Purpose:** Unpack manually-downloaded batch output files and parse responses.

**Optional:** Skip if using `02_simulate.py --mode async` or `--download BATCH_ID`.

**Input:**
- `Data/Simulation/Batch_Output/*.jsonl` (downloaded from OpenAI dashboard)

**Output:**
- `Data/Simulation/aggregate_simulation_raw_{cfg}.jsonl` (same format as async output)

### Usage

```bash
# Unpack batch output for no_reasoning config
python 03_unpack_batches.py --config no_reasoning

# Specify custom input directory
python 03_unpack_batches.py --config no_reasoning --input-dir /path/to/batch/output
```

### Workflow

1. Submit batches via `02_simulate.py --generate-only` + manual OpenAI dashboard upload
2. Wait for batches to complete (typically 1–2 hours)
3. Download *.jsonl files from dashboard → save to `Data/Simulation/Batch_Output/`
4. Run `03_unpack_batches.py` to consolidate and parse

Each batch output line contains:
- `custom_id` — request identifier
- `response.body.choices[0].message.content` — LLM response text
- Parsed → {seq_id, arm_id, outcome_id, value, parse_ok}

---

## 04_compare_effects.py

**Purpose:** Compute LLM-predicted treatment effects and compare against ground truth.

**Input:**
- `Data/Ground_Truth/study_data.jsonl` (design + GT effects)
- `Data/Simulation/aggregate_simulation_raw_{cfg}.jsonl` (per-respondent LLM responses)

**Output:**
- `Data/Results/effects_table_{cfg}.csv` — detailed comparison table
- `Data/Results/effects_summary_{cfg}.txt` — summary statistics

### Treatment Effect Computation

For each study × treatment_arm × outcome:

```
LLM_effect = mean(LLM responses in treatment_arm)
           − mean(LLM responses in control_arm)

Observed_effect (from PDF) = GT_delta
```

If either arm is missing LLM responses, the contrast is marked `comparable=False`.

### Output: effects_table_{cfg}.csv

Columns:
- `seq_id`, `study_label`, `arm_id`, `outcome_id`, `outcome_name`
- `control_arm` — which arm was the control
- `gt_delta` — observed treatment effect
- `llm_effect` — predicted treatment effect
- `gt_treatment_mean`, `gt_control_mean` — raw GT values
- `gt_n_treatment`, `gt_n_control` — sample sizes
- `metric` — "mean", "proportion", "coefficient", "other"
- `comparable` — True if both LLM control & treatment means exist
- `note` — reason if not comparable

### Output: effects_summary_{cfg}.txt

Example:
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
  seq=103  Study Title Here                            n= 8  r=+0.654  p=0.027  sign=100%
  seq=109  Another Study                               n= 6  r=+0.340  p=0.480  sign=67%
  ...
```

### Usage

```bash
# All studies
python 04_compare_effects.py --config no_reasoning

# Sound studies only (excludes seq 151, 164, 169, 174, 176)
python 04_compare_effects.py --config no_reasoning --sound-only
```

### Metrics

- **Pearson r:** Correlation between predicted and observed effects
- **Spearman r:** Rank correlation (robust to outliers)
- **RMSE:** Root mean squared error of predictions
- **Sign accuracy:** % of non-zero effects predicted with correct sign
- **Fisher z-weighted r:** Meta-analytic r accounting for study sample size

---

## 05_plot.py

**Purpose:** Generate two-panel PDF figure: scatter of predicted vs observed effects + per-study r bars.

**Input:**
- `Data/Results/effects_table_{cfg}.csv` (output from script 04)

**Output:**
- `Figures/effects_{cfg}.pdf` — publication-ready figure

### Figure Layout

**Top panel:** Scatter plot
- X-axis: GT treatment effect (Δ_GT)
- Y-axis: LLM predicted effect (Δ_LLM)
- Points: one per study × treatment arm × outcome; colored by study
- Diagonal line: perfect prediction (Δ_LLM = Δ_GT)
- OLS fit line: actual trend
- Legend: study labels with Pearson r and p-value
- Title includes: n = # contrasts, sign accuracy %

**Bottom panel:** Per-study r bars
- X-axis: study seq_id (n = # comparable contrasts in parens)
- Y-axis: within-study Pearson r (−1 to +1)
- Bar color: study color from top panel
- Hatched bars: studies with < 2 contrasts or zero variance (r undefined)
- Annotations: Fisher z-weighted r + # studies with r

### Usage

```bash
# Generate figure for no_reasoning config
python 05_plot.py --config no_reasoning

# Sound studies only
python 05_plot.py --config no_reasoning --sound-only

# All three configs
for cfg in no_reasoning reasoning_low reasoning_medium; do
  python 05_plot.py --config $cfg
done
```

### Styling

- matplotlib backend: Agg (no display window)
- Figure size: 13" × 11" (full page)
- Color palette: 16-color cycle (repeats if > 16 studies)
- Fonts: 6.5–12pt depending on element

---

## Data Flow Diagram

```
PDFs
  ↓ [01]
study_data.jsonl
  ├→ instrument (preamble, arms, outcomes)
  └→ ground_truth.effects (GT Δ values)
        ↓ [02 or 02→03]
  aggregate_simulation_raw_{cfg}.jsonl (per-respondent responses)
        ↓ [04]
  effects_table_{cfg}.csv (predicted Δ vs GT Δ)
        ↓ [05]
  effects_{cfg}.pdf (scatter + bars)
```

---

## Environment Setup

### API Keys

```bash
export OPENAI_API_KEY="sk-..."
```

Required for scripts 01 and 02. Store in ~/.zshrc or ~/.bashrc for persistence.

### Dependencies

Install via pip:
```bash
pip install openai pdfplumber numpy scipy matplotlib tqdm
```

or

```bash
pip install -r requirements.txt
```

(requirements.txt not included; adjust based on your environment)

### Python Version

Tested on Python 3.10+. Uses f-strings, type hints, and async/await.

---

## Common Workflows

### Extract a few studies first

```bash
python 01_extract_study_data.py --seq-ids 12 19 30
```
Validates extraction quality before running all 36 studies.

### Test simulation quality before full run

```bash
python 02_simulate.py --mode async --config no_reasoning --n 10
python 04_compare_effects.py --config no_reasoning
python 05_plot.py --config no_reasoning
```
Quick feedback loop; n=10 per arm = small output.

### Compare reasoning effort impact

```bash
python 02_simulate.py --generate-only --all-batches --n 50
# [submit all three configs to OpenAI]
# [download all three when complete]
python 03_unpack_batches.py --config no_reasoning
python 03_unpack_batches.py --config reasoning_low
python 03_unpack_batches.py --config reasoning_medium

for cfg in no_reasoning reasoning_low reasoning_medium; do
  python 04_compare_effects.py --config $cfg
  python 05_plot.py --config $cfg
done
# Compare effects_{no_reasoning,reasoning_low,reasoning_medium}.pdf
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| "No PDF found" | PDF name doesn't match seq_id | Rename to {seq_id}.pdf |
| "OPENAI_API_KEY not set" | Missing env variable | `export OPENAI_API_KEY="sk-..."` |
| "Unsupported parameter: 'max_tokens'" | Old script with gpt-5.4 | Update to use max_completion_tokens |
| "Rate limited" | Too many concurrent requests | Script auto-retries; be patient |
| "400 - invalid_request_error" | Batch file is malformed | Regenerate with 02_simulate.py --generate-only |
| "Parse failures > 10%" | LLM responses don't match expected format | Check response_instruction clarity; may need parser tuning |
| "No comparable contrasts" | Missing LLM control or treatment arm data | Check which arms have responses in aggregate_simulation_raw_{cfg}.jsonl |

---

## Questions?

See the main [README.md](../README.md) for project overview and [METHODOLOGY.md](../METHODOLOGY.md) for design rationale.
