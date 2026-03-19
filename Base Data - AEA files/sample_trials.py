import json
import random

INPUT_FILE = "trials_filtered.json"
OUTPUT_FILE = "trials_sampled_50.json"
N = 50
SEED = 42

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        trials = json.load(f)

    keys = list(trials.keys())

    if len(keys) < N:
        raise ValueError(f"Only {len(keys)} trials available, cannot sample {N}")

    random.seed(SEED)
    sampled_keys = random.sample(keys, N)

    sampled = {k: trials[k] for k in sampled_keys}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"Sampled {N} trials")


if __name__ == "__main__":
    main()

