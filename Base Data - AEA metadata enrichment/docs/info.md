## 1) Ingest the AEA registry → create trial records

You start with the AEA RCT Registry JSON. For each registry entry you keep:

* identifiers (`rct_id`, `doi_url`, title)
* timing fields (start/end, intervention start/end)
* the big text blobs (`intervention_text`, `experimental_design`)
* whatever structured-ish registry fields exist (randomization unit/method, outcomes, sample sizes, keywords)

Then you filter to the trials you want (e.g., status completed, has DOI/link, etc.). This gives you a clean `trials_filtered.json` and optionally a sampled subset for testing.

## 2) Build a “design spec” from registry text (metadata-level)

You convert each trial into a single machine-readable object (your JSONL row) that includes:

* normalized metadata
* deduped outcomes
* a structured representation of assignment:

  * either **arms** (mutually exclusive assignment conditions)
  * and/or **factors** (cross-cutting dimensions that generate arms)
* plus `assignment_rules` in plain English
* plus `evidence_quotes[]` that *prove* every structured claim by pointing to verbatim text

This stage is basically: “turn messy registry prose into a schema the rest of the pipeline can rely on.”

## 3) Enrich bibliographic info with OpenAlex (find the paper)

Registry metadata is often insufficient to reliably find the paper PDF. So you query OpenAlex:

* If DOI exists: lookup by DOI → get work record + OA links + alternate locations
* If no DOI: search by title/authors → fuzzy match best candidate

Output is a per-trial “source plan” that lists which URLs to try in what order (OA PDF first, then alternates, then working paper sources, etc.).

## 4) Acquire the document (PDF/HTML) with full logging

You download the paper (preferably PDF). For every attempt you log:

* attempted URL
* success/failure
* content type, bytes, sha256
* where it was saved
* errors (never silent)

Nothing is overwritten; everything is auditable.

## 5) Normalize text (so extraction is deterministic-ish)

You convert the acquired document to text:

* PDF → `pdftotext` (or similar) with page markers retained
* HTML → boilerplate-stripped main text

You store the text in a predictable location (`papers_text/{rct_id}.txt`) and save metadata about extraction.

## 6) LLM design extraction (registry + paper) into the *same schema*

Now you run the LLM against:

* registry text blocks
* paper full text (if available)

And it outputs the same schema you already defined, but “better” because it can use the full paper. Key properties:

* strict JSON schema mode
* temperature 0
* requires evidence quotes for every arm/factor/assignment claim
* validation checks: quotes must appear in source text, no hallucinated arms, etc.

This produces `design_specs_enriched.jsonl` with flags like:

* `llm_validation_passed`
* `design_completeness`
* `needs_manual`

At this point you have a defensible representation of “what was randomized, to whom, and what outcomes mattered.”

## 7) LLM results extraction (hard mode)

Using paper text, you extract empirical results in a structured way:

* outcome name
* treatment vs comparator (or factor contrasts)
* estimate + uncertainty (SE/CI/p)
* sample size used
* table/figure reference
* verbatim evidence quote anchoring the number

Same strict schema + quote validation. Output: `results_structured.jsonl`.

## 8) Harmonize across studies (so comparisons are possible)

You standardize things so apples-to-apples evaluation works:

* outcome naming normalization
* consistent sign conventions
* identify “main” effect vs interactions in factorial designs
* map treatment contrasts correctly (especially when you have factors generating arms)

This produces a clean dataset of “design + ground-truth effects.”

## 9) Simulation: feed design into LLM-as-agent (Homo Silicus)

For each study you create a simulation prompt that includes:

* treatment assignment structure (arms/factors)
* population/context
* intervention description
* outcome definitions

Then the LLM outputs predictions:

* direction of effect (sign)
* maybe magnitude bins
* optional simulated micro-level behavior (if you go that route)

Output: `prediction_outputs.jsonl`.

## 10) Evaluate predictions vs empirical reality

You compute metrics like:

* sign accuracy (% correct direction)
* calibration / confidence consistency (if you collect confidence)
* correlation with effect sizes (if you extract standardized estimates)
* heterogeneity by domain/context/design type

This is where you get the “does the model reproduce known RCT patterns” headline results.

## 11) Auditability is a first-class product

For every trial you can trace:
registry snapshot → OpenAlex record → download attempts → raw PDF/HTML → extracted text → LLM output → validation logs → final structured JSON.

That’s what makes the pipeline credible: it’s not just “LLM said so,” it’s “LLM said so and here is the exact supporting text.”

---

### How the schema fixes matter

Those fixes make the pipeline *stable*:

* one clustering variable avoids downstream branching bugs
* correct arm roles prevents false “treatment vs control” assumptions in evaluation
* non-null randomization unit makes sampling/SE rules consistent
* verbatim evidence quotes make validation meaningful
* removing stale `trial_card` avoids humans/LLMs reading contradictory summaries

If you want, I can rewrite this as a 10-line “PI-ready” explanation or as a developer checklist (“inputs/outputs per stage + failure modes”).
