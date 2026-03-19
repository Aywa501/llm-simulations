import json
from pathlib import Path

INPUT_FILE = "trials.json"
OUTPUT_FILE = "trials_filtered.json"

def is_nonempty_string(x):
    return isinstance(x, str) and x.strip() != ""

def eligible(trial):
    # A. Completed
    if trial.get("Status") != "Completed":
        return False

    # B. Core text
    if not is_nonempty_string(trial.get("Abstract")):
        return False
    if not is_nonempty_string(trial.get("Primary Outcomes (end points)")):
        return False

    # C. Some treatment description
    has_intervention = is_nonempty_string(trial.get("Intervention(s)"))
    has_design = is_nonempty_string(trial.get("Experimental Design"))
    if not (has_intervention or has_design):
        return False

    # D. Sample size exists
    if not is_nonempty_string(trial.get("Sample size: planned number of observations")):
        return False

    return True


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        trials = json.load(f)

    # trials is a dict keyed by "0", "1", ...
    filtered = {
        k: v for k, v in trials.items() if eligible(v)
    }

    print(f"Kept {len(filtered)} / {len(trials)} trials")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

