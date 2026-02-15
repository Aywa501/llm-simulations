# Automated RCT Design & Result Extraction Pipeline

## High-Level Objective
To systematically extract machine-readable experimental design specifications and empirical results from Economics RCT Registry entries (and eventually full papers) to power downstream simulation agents ("Homo Silicus").

## Core Pipeline Architecture

### 1. Ingestion & Filtering
*   **Source**: AEA RCT Registry JSON.
*   **Action**: detailed parsing of registry entries, filtering for completed studies with DOIs/links.
*   **Output**: `trials_filtered.json`

### 2. Design Extraction (Canonical Schema v3.1)
*   **Goal**: robust, audit-proof representation of "what was randomized and how".
*   **Engine**: `enrich_design_specs_llm.py` using `gpt-5.2` (or similar).
*   **Key Innovation**: A "Strict Canonical Schema" that handles complex designs without data loss.

#### The Canonical Schema (v3.1) semantics:
*   **`design_type`**: one of [`simple_multiarm`, `factorial`, `cluster_rct`, `encouragement`, ...].
*   **`is_clustered`**: Boolean. Strictly `True` if and only if randomization unit is not "individual participant".
*   **`arms`**:
    *   List of explicit assignment conditions (cells).
    *   **Rule**: Must be populated even for factorial designs if cells are named (e.g. "T1", "Control", "Barg1").
    *   **Roles**: [`control`, `treatment`, `experimental`, `active_comparator`, `placebo`].
*   **`factors`**:
    *   List of orthogonal dimensions (e.g. "Price", "Incentive").
    *   Used mainly for `factorial` designs.
*   **`evidence_quotes`**:
    *   **Strict Anchoring**: Every claim in `arms` or `factors` must link to a verbatim quote ID (e.g. `["eq1"]`).
    *   Prevents hallucinations and allows human auditing.
*   **`unit_of_randomization_canonical`**: Standardized string (e.g. "school", "individual").

### 3. Verification & Validation
*   **Logic**: `validate_extraction` function in Python.
*   **Checks**:
    *   All quote IDs typically exist.
    *   Quotes match source text verbatim.
    *   Design consistency (e.g. Factorial must have factors).
*   **Fallback**: Retries with "Strict Mode" prompt if validation fails.

### 4. Downstream (Planned/In-Progress)
*   **Paper Acquisition**: usage of OpenAlex to find and download PDFs.
*   **Results Extraction**: using the design spec to target specific outcomes and contrasts in the full text.
*   **Harmonization**: standardizing outcomes for cross-study comparison.
*   **Simulation**: Agents predicting results based on the design spec.

## Usage Guide for Agents
When working with this project, assume:
1.  **Data Truth**: `design_specs_enriched.jsonl` contains the ground-truth design.
2.  **Schema Authority**: The schema in `enrich_design_specs_llm.py` is the single source of truth.
3.  **Ambiguity Policy**: The system prioritizes *correctness* over *completeness*. (e.g. `analysis_unit` is `null` if not explicit, rather than guessed).
