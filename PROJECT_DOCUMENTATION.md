# LLM Simulations Project Documentation

## 1. Project Overview

**Project Name**: `llm-simulations`

**Goal**: This project aims to extract, enrich, and validate experimental design specifications from AEA RCT Registry entries using Large Language Models (LLMs). The structured data is intended for use in downstream simulations.

**Key Capabilities**:
- **Scraping**: Automated retrieval of trial documents (PDFs, HTML) from the AEA RCT Registry.
- **Data Normalization**: Converting raw registry data into a standardized "Design Spec" format.
- **LLM Enrichment**: Using OpenAI's models to infer complex experimental design details (e.g., factorial structures, randomization units) and anchoring them with evidence quotes.
- **Quality Assurance**: Automated validation of LLM-extracted claims against the source text.
- **Text Metrics**: logical evaluation of text quality using advanced NLP metrics (BERTScore, BLEURT, etc.).

---

## 2. Directory Structure

The project is organized into three main components:

```
llm-simulations/
├── README.md               # Project entry point
├── PROJET_DOCUMENTATION.md # Detailed documentation (this file)
├── text_metrics.ipynb      # Notebook for analyzing text generation quality
├── Data Retrival/          # Core pipeline for design spec extraction
│   ├── commands.sh         # Helper script (env vars)
│   └── Design spec/        # Main Python scripts for the pipeline
│       ├── build_design_specs.py       # Step 1: Normalize raw data
│       ├── enrich_design_specs_llm.py  # Step 2: LLM Enrichment
│       ├── unpack_batch_results.py     # Step 3: Process batch results
│       ├── utils.py                    # Shared utilities
│       └── data/                       # Input/Output data directory
└── project/                # Scraping and raw data collection
    ├── scrape_sources.py   # Extract URLs from registry data
    ├── scrape_download.py  # Download artifacts (PDFs, HTML)
    └── artifacts/          # Downloaded trial documents (by RCT ID)
```

---

## 3. Workflow & Usage

### Prerequisites
- Python 3.8+
- OpenAI API Key (for enrichment)
- Dependencies: `pandas`, `numpy`, `nltk`, `spacy`, `openai`, `bert-score`, `bleurt`, `moverscore` (see `text_metrics.ipynb` for details).

### Phase 1: Data Collection (Scraping)
Located in `project/`.

1.  **Extract Sources**:
    Parse the raw trials JSON to find relevant URLs (papers, data links).
    ```bash
    python project/scrape_sources.py
    ```
    *Input*: `trials_sampled_50.json` (assumed path)
    *Output*: `sources_50.jsonl`

2.  **Download Artifacts**:
    Fetch the identified resources.
    ```bash
    python project/scrape_download.py
    ```
    *Input*: `sources_50.jsonl`
    *Output*: `project/artifacts/<rct_id>/*` and `project/scrape_index.jsonl`

### Phase 2: Design Specification Pipeline
Located in `Data Retrival/Design spec/`.

1.  **Build Initial Specs**:
    Convert the raw registry JSON into a clean, flat JSONL format.
    ```bash
    python "Data Retrival/Design spec/build_design_specs.py" \
      --in "Data Retrival/Design spec/data/trials_sampled_50.json" \
      --out "Data Retrival/Design spec/data/design_specs.jsonl"
    ```

2.  **Enrich with LLM**:
    Use OpenAI to extract structured design info (arms, factors, randomization).
    *Standard Mode (Synchronous)*:
    ```bash
    python "Data Retrival/Design spec/enrich_design_specs_llm.py" \
      --in "Data Retrival/Design spec/data/design_specs.jsonl" \
      --out "Data Retrival/Design spec/data/design_specs_enriched.jsonl" \
      --model "gpt-4o"
    ```
    *Batch Mode*:
    Set `USE_BATCH_API = True` in the script to generate a batch input file instead.

3.  **Unpack Batch Results**:
    If using the Batch API, process the results and merge them back.
    ```bash
    python "Data Retrival/Design spec/unpack_batch_results.py" \
      --batch-dir "Data Retrival/Design spec/data/batch_outputs" \
      --output "Data Retrival/Design spec/data/design_specs_enriched.jsonl"
    ```

### Phase 3: Analysis
1.  **Text Metrics**:
    Open `text_metrics.ipynb` to run textual analysis on the generated or retrieved text. This notebook calculates:
    - Syntactic Complexity (via spaCy)
    - Lexical Richness (TTR, Hapax Rate)
    - Semantic Similarity (BERTScore, BLEURT, MoverScore)

---

## 4. Key Script Details

### `enrich_design_specs_llm.py`
This is the core logic for intelligence extraction.
- **Prompting**: logic to construct prompts from trial text.
- **Validation**:
    - Checks if extracted `evidence_quotes` actually exist in the source text (fuzzy matching).
    - Validates consistency between `arms` and `factors`.
    - Enforces schema constraints (e.g., factorial designs must have factors).
- **Caching**: Uses `.llm_cache_design_extract.json` to avoid redundant API calls.

### `scrape_download.py`
- **Smart Fetching**: Checks headers to identify PDFs even if the URL doesn't end in `.pdf`.
- **Discovery**: Parses HTML to find linked PDFs (e.g., "Download PDF" buttons).
- **Rate Limiting**: Includes sleeps and timeouts to be polite to servers.

---

## 5. Data Schema (`design_specs_enriched.jsonl`)

The final output is a JSONL file where each line is a JSON object with:

- **`rct_id`**: Unique identifier.
- **`provenance`**: Original data from the registry.
- **`enrichment`**:
    - **`llm`**: Metadata about the extraction (model, prompt version, fingerprint).
    - **`derived`**: The structured design extracted by the LLM (Design Type, Clustering, Arms, Factors).
    - **`evidence`**: Quotes supporting the derived structure.
    - **`quality`**: Validation flags (passed/failed, specific error messages).

