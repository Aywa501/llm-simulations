import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parents[1] / "Data" / "Ground_Truth"

# Based on previous robust text-based simulatability assessment
CLEARLY_SIMULATABLE = {19, 96, 103, 118, 125, 131, 141, 143, 145, 150, 151, 152, 159, 160, 162, 164, 169, 174, 176, 178, 183}

# Borderline studies require real-world behaviors or visual stimuli
BORDERLINE = {30, 72, 108, 135, 148, 157, 170}

def main():
    parser = argparse.ArgumentParser(description="Filter studies to only those that can be safely simulated by LLMs.")
    parser.add_argument("--include-borderline", action="store_true",
                        help="Include borderline studies (e.g., those requiring visual intuition or measuring real-world gym visits)")
    args = parser.parse_args()

    # Automatically resolve the latest complete extraction file
    pass2_path = DATA_DIR / "study_enriched_aggregate_pass2.jsonl"
    pass1_path = DATA_DIR / "study_enriched_aggregate.jsonl"
    in_path = pass2_path if pass2_path.exists() else pass1_path

    if not in_path.exists():
        print(f"Error: Extraction files not found in Ground_Truth dir. Run extraction first.")
        return

    out_path = DATA_DIR / "simulatable_studies.jsonl"
    
    valid_ids = CLEARLY_SIMULATABLE.copy()
    if args.include_borderline:
        valid_ids.update(BORDERLINE)

    retained = []
    dropped = []

    with open(in_path, "r") as f:
        for line in f:
            if not line.strip(): 
                continue
            rec = json.loads(line)
            seq_id = rec["seq_id"]
            
            # Structural constraint checks
            instrument = rec.get("instrument", {})
            has_arms = len(instrument.get("treatment_variations", [])) > 0
            has_outcomes = len(instrument.get("outcome_questions", [])) > 0
            
            if not instrument.get("found"):
                dropped.append((seq_id, "No coherent survey instrument could be established"))
                continue
            if not has_arms or not has_outcomes:
                dropped.append((seq_id, "Missing experimental arms or outcomes structurally"))
                continue

            # Assessment checks
            if seq_id in valid_ids:
                retained.append(rec)
            elif seq_id in BORDERLINE:
                dropped.append((seq_id, "Flagged as borderline (visual stimuli/real-world behavior) - use --include-borderline to override"))
            else:
                dropped.append((seq_id, "Failed manual simulatability assessment (requires physical action, kids, standardized tests, etc.)"))

    with open(out_path, "w") as f:
        for rec in retained:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Loaded records from: {in_path.name}")
    print(f"---------------------------------------------------")
    print(f"Total read     : {len(retained) + len(dropped)}")
    print(f"Retained       : {len(retained)} (Successfully mapped for simulation)")
    print(f"Dropped        : {len(dropped)}")
    print(f"Output saved to: {out_path.name}")
    print(f"---------------------------------------------------")
    
    if dropped:
        print("\nExcluded Studies:")
        for seq_id, reason in sorted(dropped):
            print(f"  seq_id={seq_id:<3} : {reason}")

if __name__ == "__main__":
    main()
