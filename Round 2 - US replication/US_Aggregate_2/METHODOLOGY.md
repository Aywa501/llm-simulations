# Methodology & Design Rationale

This document explains why US_Aggregate_2 is structured the way it is, and how it differs from earlier approaches.

## Core Research Question

**Can LLMs accurately predict how human participants will respond to behavioral economics experiments?**

More precisely: Given a treatment prompt and outcome question, can an LLM simulate human responses closely enough that the **direction and magnitude of treatment effects** match empirical observations?

---

## Treatment Effects vs Arm Means

### The Hewitt et al. Study

Hewitt et al. (2024) demonstrated that LLMs can predict the **direction** of treatment effects across 109 psychology/behavioral economics experiments. Their metric: correlation between observed and predicted treatment effects (Δ).

**Treatment effect definition:**
```
Δ = mean(treatment arm) − mean(control arm)
```

This is the causal quantity of interest in RCTs. By predicting Δ rather than raw arm means, the LLM must capture:
1. The absolute effect magnitude (how big is the shift?)
2. The direction (does the treatment increase or decrease the outcome?)
3. The base rate (what's the control arm baseline?)

### Why Not Raw Arm Means?

An earlier iteration of this pipeline compared raw arm means:
```
r = corr(LLM_mean(arm_i), human_mean(arm_i)) for all arms across all studies
```

This approach has a critical flaw:

**Scale confounding.** If a study's outcome scale is 1–100, all responses will be shifted up. If another study's scale is 0–1, all responses will be lower. Cross-study correlations are dominated by scale differences, not by LLM fidelity.

**Example:** Study A (scale 1–7) has mean arm response 5.0. Study B (scale 0–100) has mean response 75. An LLM that predicts "5" for both gets r ≈ 1.0 for Study A (on scale) but r ≈ 0 for Study B (off scale). The overall correlation is inflated by Study A's inflated apparent prediction accuracy.

**Study size dominance.** One study (seq=174) produced 105/193 comparable arm-outcome pairs (55%). Its effect sizes (or lack thereof) dominated the overall correlation, obscuring patterns in other studies.

### The Solution: Treatment Effects

Treatment effects are **scale-independent** (relative, not absolute):
```
Δ = (mean_T − mean_C) / (scale range)  [implicitly scaled]
```

If a treatment shifts responses from 3 to 3.5 on a 1–7 scale, that's a Δ of +0.5. If another treatment shifts from 50 to 60 on a 0–100 scale, that's also a Δ of +0.5 (relative). Comparing these directly is fair.

**Within-study r** (studied in earlier iterations with z-scoring) showed this explicitly:
- Without normalization: overall r ≈ 0.2, dominated by seq=174
- With z-scoring: overall r ≈ 0.4, more balanced across studies
- Comparing effects directly: r ≈ 0.43, cleaner and theoretically justified

**Advantages:**
1. **Methodologically sound:** Matches Hewitt et al. definition of "successful prediction"
2. **Study-balanced:** No single study dominates due to effect size or scale
3. **Conceptually clear:** "Did the LLM predict the direction and magnitude of the treatment?"
4. **Handles varied designs:** Naturally accommodates studies without single control arms (simply exclude from analysis)

---

## Two-Pass Extraction Design

Why extract design (Pass 1) and results (Pass 2) separately?

### Motivation

Early extraction attempts tried to extract everything in one pass:
- Design: arms, outcomes, controls
- Results: Δ values, sample sizes, metrics

This produced high error rates and inconsistent coverage (< 50% of expected effects).

### The Two-Pass Solution

**Pass 1 — Design:**
- LLM reads the paper and returns the experimental **structure**
- Arm labels (e.g., "Control", "Treatment A")
- Outcome names (e.g., "Purchase Intent")
- Outcome scale/format (e.g., "1–7 Likert")
- Which arm is control

**Pass 2 — Results:**
- LLM uses exact **arm_ids and outcome_ids from Pass 1** to extract effect values
- No string matching needed; LLM uses the structure we've already agreed on
- Extraction task is narrower: "Find the Δ for [specific arm_id] × [specific outcome_id]"

### Key Insight: Shared IDs

The critical innovation is that both passes generate **shared arm_ids and outcome_ids**:

Pass 1 produces:
```json
{
  "treatment_variations": [
    {"arm_id": "treatment_a", "arm_label": "Treatment A", ...}
  ],
  "outcome_questions": [
    {"outcome_id": "purchase_intent", "outcome_name": "Purchase Intent", ...}
  ]
}
```

Pass 2 uses those exact IDs:
```json
{
  "effects": [
    {
      "arm_id": "treatment_a",
      "outcome_id": "purchase_intent",
      "delta": 0.45
    }
  ]
}
```

**Downstream benefit:** Script 04 (compare_effects.py) can join `instrument` and `ground_truth.effects` via exact key match. No fuzzy string matching, no ambiguity.

### Coverage Grading

- **≥ 75% coverage:** status="ok" (complete extraction; fully usable)
- **25–75% coverage:** status="partial" (some effects found; usable with caveats)
- **< 25% coverage:** status="failed" (triggers automatic retry; if still low, likely paper ambiguity)

This three-tier grading acknowledges that some studies have ambiguous or incomplete results reporting, but still allows us to use partial data.

---

## Unified Data Schema

### study_data.jsonl Structure

One JSON line per study with **unified schema:**

```json
{
  "seq_id": 103,
  "instrument": { ... },
  "ground_truth": { "effects": [...] }
}
```

**Why unified?**

Earlier pipelines had separate files:
- `simulatable_studies.jsonl` (design only)
- `study_effects_gt.jsonl` (effects only)
- Different arm_id/outcome_id values in each file (extraction run separately)

This forced fuzzy matching:
- Does "Treatment 1" from design match "Treat_1" from effects?
- Do both refer to the same experimental arm?

**Fuzzy matching is error-prone.** Results showed ~15% of rows dropped due to arm/outcome mismatches.

### The Fix: Single Extraction, Shared IDs

Both instrument and effects are extracted from the same PDF in the same calls. They use the same slugified IDs. Results in **zero matching errors** downstream.

---

## LLM Model Choices

### Extraction (script 01): gpt-5.4 + medium reasoning

**Why reasoning?**
- Extraction requires parsing complex research papers
- Identifying control groups, understanding scale ranges, extracting statistical tables
- Medium reasoning provides depth without extreme latency

**Why gpt-5.4 (not 4.1)?**
- Better at understanding research methodology and statistical concepts
- More reliable JSON output (rarely drops fields)
- Reasoning helps disambiguate papers written in varied formats

**Cost/latency trade-off:**
- ~90 seconds per study (reasonable for one-time extraction)
- Medium reasoning is slower than base gpt-4.1 but faster than o1

### Simulation (script 02): gpt-4.1, no reasoning

**Why no reasoning?**
- Participant simulation is **stateless** — each response is independent
- Reasoning adds latency with minimal benefit
- Need fast batch processing for thousands of responses

**Why gpt-4.1?**
- Sufficient for role-playing (participant in survey)
- Fast enough for batch API
- Cost-effective for large n

**Sampling strategy:**
- `no_reasoning`: temperature=1, top_p=1 (diverse, untempered)
- `reasoning_low/medium`: not used for simulation (incompatible with temperature)

---

## Why This Pipeline Matters

### Previous Approach Issues

1. **Hardcoded control arms:** Earlier versions hardcoded which arm was the control for each study. User feedback: "I wouldn't trust these — validate against the papers themselves."

2. **Fuzzy matching failures:** Arm/outcome mismatches meant ~20% of potential comparisons were lost.

3. **Scale dominance:** Raw arm mean correlations were dominated by large-effect or large-scale studies.

4. **Unclear simulatability:** No systematic assessment of which studies were actually suitable for LLM simulation.

### This Version's Improvements

1. **LLM-verified controls:** Extraction reads the paper and identifies controls; no hardcoding.

2. **Shared IDs:** No fuzzy matching; exact key joins.

3. **Treatment effects:** Scale-independent, study-balanced, methodologically sound.

4. **Explicit simulatability:** Pass 1 classifies each study as simulatable/not with a note. Studies with visual tasks, audio, real-money transactions, etc. are flagged.

5. **Caching & retry:** Expensive API calls are cached; low coverage automatically retries; safe to re-run.

---

## Methodological Decisions

### What Counts as "Comparable"

A study × arm × outcome comparison is "comparable" if:
1. Ground truth has a delta value (extracted from paper)
2. LLM has responses for both control and treatment arms
3. Both have ≥ 1 valid parsed response

If either condition fails, the row is marked `comparable=False` and excluded from correlation statistics.

### Sign Accuracy Metric

Sign accuracy answers: "Did the LLM predict the direction of the effect?"

```
sign_acc = # (where sign(Δ_LLM) == sign(Δ_GT)) / # (where Δ_GT ≠ 0)
```

This is a **conservative metric**. An LLM that predicts Δ = 0.001 when truth is Δ = 0.5 scores as "correct direction" but has huge error in magnitude. Sign accuracy only checks direction.

### Fisher z-Transform

Per-study correlations are averaged using Fisher z-transform:

```
z_i = arctanh(r_i)  [z-transform each r]
z_avg = Σ(n_i × z_i) / Σ(n_i)  [weighted average]
r_avg = tanh(z_avg)  [inverse transform]
```

**Why?** Pearson r is not normally distributed. z-transform stabilizes variance. Weighting by n gives more influence to studies with more contrasts (more stable estimates).

This is the meta-analytic standard for combining effect sizes.

---

## Unsound Studies

Five studies are excluded under `--sound-only` flag due to fundamental simulatability issues:

| Study | Reason | Detail |
|-------|--------|--------|
| 151 | Visual task | Matrix puzzle — LLM cannot see the image |
| 164 | Audio outcomes | Requires listening to recordings |
| 169 | Money/choice collapse | Real-money dictator game; binary yes/yes is not a real choice |
| 174 | Missing question text | All 15 outcomes have `question_text = null` |
| 176 | Identical arms | All treatment arms have identical text; no variation |

These studies should **not** be included in correlation analyses, as they involve fundamental limitations of LLM simulation (can't see images, can't hear audio) or data quality issues (missing question text).

---

## Scalability & Performance

### Concurrency Limits

- **Extraction:** 4 concurrent API calls (semaphore)
- **Simulation (async):** 4 concurrent API calls (semaphore)
- **Simulation (batch):** No concurrency limit (OpenAI queues batches)

Why 4? A safe limit that avoids rate-limiting (OpenAI default: 500 req/min for some tiers). 4 concurrent calls ≈ 4–5 requests per second ≈ 240–300 per minute (safe margin).

### Caching

Extraction results are cached in `.extract_study_data_cache.json` with keys:
- `p1__{seq_id}__{model}` — Pass 1 cached design
- `p2__{seq_id}__{model}` — Pass 2 cached effects

This avoids re-calling the API for studies already extracted (even if model version changes slightly).

**To force re-extraction:** Use `--force` flag (deletes cache entries as it runs).

---

## Alternative Approaches Considered

### 1. Single-Pass Extraction
**Rejected because:** Low coverage (< 50% of expected effects). Two passes with shared IDs are more reliable.

### 2. Fuzzy String Matching (arm_id matching)
**Rejected because:** ~15% error rate; impossible to debug without manual review. Shared IDs are deterministic.

### 3. Within-Study Z-Score Normalization
**Attempted in earlier version; replaced by treatment effects because:**
- More complex pipeline (needed normalization step)
- Still requires arm means (includes design-carrying information)
- Treatment effects are simpler and theoretically cleaner
- Normalization was a workaround for scale dominance; effects solve it directly

### 4. Micro-Batch Processing
**Rejected because:** Overhead of chunking, unpacking multiple files. Better to either:
- Use async mode for small runs (direct output)
- Use batch API for large runs (OpenAI handles queuing)

---

## Future Extensions

1. **Heterogeneous effect moderation:** Do LLMs predict some demographics/conditions better than others?

2. **Reasoning effort comparison:** What's the marginal benefit of reasoning for simulation? (Pipeline supports this: generate all three reasoning configs, compare outputs.)

3. **Cross-domain generalization:** Train on one dataset, test on another. Requires multiple study pools.

4. **Fine-tuning:** Can we improve LLM participant simulation by fine-tuning on observed human responses?

5. **Explicit role-playing:** Vary system prompts (e.g., "You are a risk-averse investor" vs "You are risk-seeking"). See if LLMs learn the role.

---

## References

**Primary reference:**
- Hewitt, J. L., et al. (2024). Predicting Results of Social Science Experiments Using Large Language Models. *Nature Human Behaviour*.

**Related literature:**
- Meta-analysis methods: Borenstein, M., et al. (2009). *Introduction to Meta-Analysis.* Wiley.
- Fisher z-transform: Fisher, R. A. (1921). On the "probable error" of a coefficient of correlation deduced from a small sample.

**LLM reasoning:**
- OpenAI docs: https://platform.openai.com/docs/guides/reasoning

---

## Questions?

See [README.md](README.md) for project overview or [Scripts/README.md](Scripts/README.md) for operational details.
