# LLM Simulations

Research code for testing whether modern LLMs can reproduce experimental results from behavioral economics and related social science studies.

At a high level, the project:

1. collects and normalizes study materials,
2. extracts experimental design and ground-truth results,
3. simulates participant responses with LLMs, and
4. compares predicted treatment effects against observed human effects.

## Repo Guide

### 1. Main active pipeline

The most up-to-date end-to-end replication workflow is in:

`Round 2 - US replication/US_Aggregate_2/`

Start here for the current treatment-effects pipeline:

- `README.md` — high-level overview and quick start
- `METHODOLOGY.md` — design rationale and evaluation choices
- `Scripts/README.md` — per-script documentation
- `Data/README.md` — data layout and file conventions

This pipeline:

- extracts study design and ground-truth effects from PDFs,
- simulates participant responses by arm and outcome,
- computes LLM-predicted treatment effects, and
- evaluates them against observed treatment effects.

### 2. Base data and metadata enrichment

The AEA registry preparation and design-spec extraction work lives in:

- `Base Data - AEA files/`
- `Base Data - AEA metadata enrichment/`

These directories contain the earlier-stage data cleaning and schema-building work used to turn registry entries into structured experimental design records.

Helpful docs:

- `Base Data - AEA metadata enrichment/docs/context_summary.md`
- `Base Data - AEA metadata enrichment/docs/info.md`

### 3. Earlier proof-of-concept work

Older exploratory simulations and trial runs live in:

- `Round 1 - 3 study trial/`

These are useful for historical context, but they are not the main current pipeline.

### 4. Deprecated material

Archived or superseded code is stored under:

- `deprecated/`

Treat this as reference material only unless you know you need it.

## Recommended Starting Points

If you want to understand the current project quickly:

1. read `Round 2 - US replication/US_Aggregate_2/README.md`,
2. read `Round 2 - US replication/US_Aggregate_2/METHODOLOGY.md`,
3. inspect `Round 2 - US replication/US_Aggregate_2/Scripts/01_extract_study_data.py` through `05_plot.py`.

If you want the registry/design-spec side:

1. read `Base Data - AEA metadata enrichment/docs/context_summary.md`,
2. inspect `Base Data - AEA metadata enrichment/Design Spec/build_design_specs.py`,
3. inspect `Base Data - AEA metadata enrichment/Design Spec/enrich_design_specs_llm.py`.

## Notes

- The repo has evolved over time, so some folders represent older iterations of the project.
- The `US_Aggregate_2` documentation is currently the best-maintained description of the active replication workflow.
- An older top-level project document was removed because it no longer matched the current repo structure or script behavior.
