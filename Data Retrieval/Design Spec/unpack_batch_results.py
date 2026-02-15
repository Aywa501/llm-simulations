import json
import argparse
import glob
import os
import sys
from typing import Dict, Any, List

# Import validation logic from the main script
# We need to add the current directory to sys.path to import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from enrich_design_specs_llm import validate_extraction, build_llm_input, stable_hash
except ImportError:
    # Fallback if running from a different directory
    print("Warning: Could not import validate_extraction. Validation will be skipped.")
    def validate_extraction(spec): return True, []

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_latest_batch_output(batch_dir: str) -> str:
    files = glob.glob(os.path.join(batch_dir, "*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {batch_dir}")
    return max(files, key=os.path.getmtime)

def extract_json_block(text: str) -> str:
    """
    Extracts the first valid JSON block (delimited by matching braces) from the text.
    Handles nested braces. If straightforward extraction fails, relies on strict JSON parsing of the substring.
    """
    text = text.strip()
    # Attempt to find the first '{'
    start = text.find('{')
    if start == -1:
        return text # Let json.loads fail
    
    # Track brace balance to find the closing '}'
    balance = 0
    end = -1
    for i in range(start, len(text)):
        char = text[i]
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1
            if balance == 0:
                end = i + 1
                break
    
    if end != -1:
        return text[start:end]
    
    # If explicit extraction fails (e.g. malformed), try to work from the end
    end = text.rfind('}')
    if end != -1:
        return text[start:end+1]
        
    return text

def main():
    parser = argparse.ArgumentParser(description="Unpack batch API results")
    parser.add_argument("--batch-dir", default="data/batch_outputs", help="Directory containing batch outputs")
    parser.add_argument("--original-input", default="data/design_specs.jsonl", help="Original input file with metadata")
    parser.add_argument("--output", default="data/design_specs_enriched.jsonl", help="Final merged output")
    args = parser.parse_args()

    # 1. Identify latest batch file
    try:
        batch_file = get_latest_batch_output(args.batch_dir)
        print(f"Processing latest batch output: {batch_file}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Load original metadata (hashed by rct_id)
    print(f"Loading original specs from {args.original_input}...")
    original_specs = {}
    try:
        raw_specs = load_jsonl(args.original_input)
        for spec in raw_specs:
            if "rct_id" in spec:
                original_specs[spec["rct_id"]] = spec
        print(f"Loaded {len(original_specs)} original specs.")
    except FileNotFoundError:
        print(f"Error: Original input file {args.original_input} not found.")
        return

    # 3. Process batch results
    print("Processing batch results...")
    enriched_results = []
    
    # Load batch data
    try:
        batch_data = load_jsonl(batch_file)
    except Exception as e:
        print(f"Error loading batch file: {e}")
        return

    successful = 0
    failed = 0

    for item in batch_data:
        rct_id = item.get("custom_id")
        if not rct_id:
            continue

        if rct_id not in original_specs:
            print(f"Warning: RCT ID {rct_id} found in batch but not in original input. Skipping.")
            continue

        original_spec = original_specs[rct_id]

        # Extract LLM validation content
        try:
            response_body = item["response"]["body"]
            
            # SUPPORT FOR /v1/responses (Structured list of messages)
            # Structure: body["output"][0]["content"][0]["text"]
            if "output" in response_body and isinstance(response_body["output"], list):
                try:
                    # We assume the last message is the assistant's reply, or the first in the list?
                    # Batch API usually returns the generated message. The list might contain just that.
                    # Based on inspection: output[0] is the message.
                    message_obj = response_body["output"][0]
                    content_list = message_obj.get("content", [])
                    if isinstance(content_list, list) and len(content_list) > 0:
                        # Assuming first content block is the text
                        message_content = content_list[0].get("text", "")
                    elif isinstance(content_list, str):
                        message_content = content_list
                    else:
                        print(f"Error: Unexpected content format in /v1/responses: {type(content_list)}")
                        failed += 1
                        continue
                except (IndexError, KeyError, TypeError) as e:
                     print(f"Error traversing /v1/responses output structure for {rct_id}: {e}")
                     failed += 1
                     continue

            # SUPPORT FOR /v1/chat/completions (legacy)
            elif "choices" in response_body:
                message_content = response_body["choices"][0]["message"]["content"]
            else:
                print(f"Error: Unknown response format for {rct_id}. Keys: {list(response_body.keys())}")
                failed += 1
                continue
            
            # OpenAI /v1/responses with strict mode returns pure JSON.
            try:
                llm_extraction = json.loads(message_content)
            except json.JSONDecodeError:
                # Fallback to extracting block if strict parse fails
                json_str = extract_json_block(message_content)
                llm_extraction = json.loads(json_str)

        except json.JSONDecodeError as e:
            print(f"Error parse JSON for {rct_id}: {e}")
            failed += 1
            continue
        except Exception as e: # Catch other potential errors during extraction
            print(f"Error processing LLM response for {rct_id}: {e}")
            failed += 1
            continue

        # --- Construct Output Object (Nested Schema) ---
        
        # 1. Provenance (Registry Metadata)
        # We need to filter for specific keys if we want to be strict, but for now copying all
        # registry fields into provenance.registry is the goal, per sample.jsonl
        # The sample.jsonl has specific fields in provenance.registry.
        # We'll copy the whole original_spec into provenance.registry for completeness/safety,
        # or we can select them. Let's copy specific ones + key leftovers to be safe?
        # The prompt implies "Raw registry fields copied verbatim".
        
        provenance = {
            "registry": original_spec.copy()
        }

        # 2. Enrichment
        
        # Derived fields from LLM
        derived = {}
        derived_keys = [
            "design_type", "unit_of_randomization_canonical", "is_clustered",
            "analysis_unit_canonical", "primary_outcomes_dedup",
            "arms", "factors", "assignment_rules", 
            "design_completeness", "extraction_sources", "notes"
        ]
        for k in derived_keys:
            if k in llm_extraction:
                derived[k] = llm_extraction[k]

        # Evidence Quotes
        evidence = {}
        if "evidence_quotes" in llm_extraction:
            evidence["quotes"] = llm_extraction["evidence_quotes"]
        else:
            evidence["quotes"] = []

        # LLM Metadata
        # Calculate fingerprint
        try:
            input_text_for_hash = build_llm_input(original_spec)
            input_fingerprint = stable_hash(input_text_for_hash)
        except Exception as e:
            print(f"Warning: Could not compute fingerprint for {rct_id}: {e}")
            input_fingerprint = "ERROR"

        llm_meta = {
            "provider": "openai",
            "model": response_body.get("model", "batch-unknown"),
            "prompt_version": "v3.1", # Hardcoded or imported? Imported would be better but simple string is safe for now.
            "input_fingerprint": input_fingerprint,
            "run_id": item.get("id"), # batch request id
            "created_at": None # Batch API doesn't give easily parseable timestamp in this view, skipping or null
        }

        # Quality (Validation)
        # Re-construct a flat object for validation (as validate_extraction expects flat extracted structure + input text)
        # validate_extraction expects 'extracted' dict (llm_extraction) and 'input_text' (str)
        
        # Validation text source: ideally the exact text used.
        # We accept 'input_text_for_hash' as the source.
        is_valid, validation_errors = validate_extraction(llm_extraction, input_text_for_hash)
        
        quality = {
            "validation_passed": is_valid,
            "validation_errors": validation_errors,
            "flags": [],
            "needs_manual": not is_valid # simplistic
        }

        # Assemble Final Object
        enriched_spec = {
            "schema_version": "design_specs_enriched.v1",
            "rct_id": rct_id,
            "provenance": provenance,
            "enrichment": {
                "llm": llm_meta,
                "derived": derived,
                "evidence": evidence,
                "quality": quality
            }
        }
        
        enriched_results.append(enriched_spec)
        successful += 1

    # 4. Write output
    print(f"Writing {len(enriched_results)} enriched specs to {args.output}...")
    with open(args.output, 'w') as f:
        for spec in enriched_results:
            f.write(json.dumps(spec, ensure_ascii=False) + "\n")

    print(f"Done. Successful: {successful}, Failed/Skipped: {failed}")

if __name__ == "__main__":
    main()
