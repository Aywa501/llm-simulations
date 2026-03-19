import json
from collections import Counter

INPUT_FILE = "trials.json"
OUTPUT_FILE = "trials_filtered.json"

def is_nonempty_string(x):
    return isinstance(x, str) and x.strip() != ""

def norm_key(k: str) -> str:
    return " ".join(str(k).strip().lower().split())

def build_key_index(trial: dict):
    idx = {}
    for k in trial.keys():
        nk = norm_key(k)
        if nk not in idx:
            idx[nk] = k
    return idx

def get_any(trial: dict, *candidates, default=None):
    idx = build_key_index(trial)
    for c in candidates:
        ck = norm_key(c)
        if ck in idx:
            return trial.get(idx[ck])
    return default

def parse_boolish(v):
    """
    Returns True/False/None (unknown).
    Accepts bool, Yes/No, True/False, TRUE/FALSE, 1/0.
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "y", "true", "t", "1"):
            return True
        if s in ("no", "n", "false", "f", "0"):
            return False
    return None

def eligible(trial):
    """
    Returns (is_eligible: bool, reason: str)
    """

    # --- 0) Public data available must be YES/TRUE ---
    public_data = get_any(
        trial,
        "Is public data available?",
        "Public data",
        "Public Data",
        "Public data available",
        "Is Public Data Available?",
    )
    public_bool = parse_boolish(public_data)
    if public_bool is not True:
        return False, "no_public_data"

    # --- 0b) If public data yes, require a URL (key name varies across exports) ---
    public_url = get_any(
        trial,
        "Public data url",
        "Public Data URL",
        "Public data URL",
        "Public data link",
        "Public Data Link",
        "Public dataset url",
        "Public dataset URL",
        "Public dataset link",
        "Public dataset URL (if yes)",
    )
    if not is_nonempty_string(public_url):
        return False, "missing_public_data_url"

    # --- A) Completed (your JSON uses 'Completed' with capital C) ---
    status = get_any(trial, "Status")
    if not (isinstance(status, str) and status.strip().lower() == "completed"):
        return False, "not_completed"

    # --- B) Core text ---
    abstract = get_any(trial, "Abstract")
    if not is_nonempty_string(abstract):
        return False, "missing_abstract"

    primary_outcomes = get_any(
        trial,
        "Primary Outcomes (end points)",
        "Primary outcome end points",
        "Primary outcome endpoints",
    )
    if not is_nonempty_string(primary_outcomes):
        return False, "missing_primary_outcomes"

    # --- C) Some treatment/design description ---
    intervention = get_any(trial, "Intervention(s)", "Intervention")
    design = get_any(trial, "Experimental Design", "Experimental design")
    if not (is_nonempty_string(intervention) or is_nonempty_string(design)):
        return False, "missing_intervention_and_design"

    # --- D) Sample size exists (planned OR final) ---
    planned_obs = get_any(trial, "Sample size: planned number of observations")
    final_obs = get_any(trial, "Final Sample Size: Total Number of Observations")
    if not (is_nonempty_string(planned_obs) or is_nonempty_string(final_obs)):
        return False, "missing_observations"

    return True, "ok"

def iter_trials(trials_obj):
    if isinstance(trials_obj, dict):
        for k, v in trials_obj.items():
            if isinstance(v, dict):
                yield k, v
    elif isinstance(trials_obj, list):
        for i, v in enumerate(trials_obj):
            if isinstance(v, dict):
                yield str(i), v
    else:
        raise TypeError(f"Unsupported trials container type: {type(trials_obj)}")

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        trials_obj = json.load(f)

    reasons = Counter()
    kept = {}

    n_total = 0
    for k, trial in iter_trials(trials_obj):
        n_total += 1
        ok, reason = eligible(trial)
        reasons[reason] += 1
        if ok:
            kept[k] = trial

    print(f"Kept {len(kept)} / {n_total} trials")
    print("Counts by reason:")
    for r, c in reasons.most_common():
        print(f"  {r}: {c}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
