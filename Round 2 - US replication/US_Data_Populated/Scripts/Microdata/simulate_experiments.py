"""
simulate_experiments.py

Participant-mode LLM simulation for 7 studies (6 green + seq=46 yellow).
Output: Data/Microdata/simulation_raw.jsonl

Note: seq=44 (Expectation Formation) excluded — ground truth accuracy scores
were computed from the actual experimental time series, which is not available.
Synthetic data would produce incomparable accuracy metrics.

Usage:
    python simulate_experiments.py [--n N] [--mode async|batch]
    python simulate_experiments.py --download BATCH_ID

Modes:
    async  (default) — real-time async API calls, progress bar
    batch            — submits OpenAI Batch API job (50% cheaper, ~minutes–hours)
                       prints batch_id on exit; use --download to retrieve results
"""

import argparse, asyncio, io, json, re, time, uuid
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from openai import AsyncOpenAI, OpenAI

DATA_DIR = Path(__file__).resolve().parents[2] / "Data" / "Microdata"
OUTPUT   = DATA_DIR / "simulation_raw.jsonl"
MODEL    = "gpt-5.4-mini"

SYSTEM_PROMPT = (
    "You are a participant in an online survey. "
    "Read the scenario and respond as an ordinary American adult. "
    "IMPORTANT: Give only the exact response format requested — "
    "a single word, a single number, or YES/NO. "
    "No explanations, no caveats, no extra text."
)

# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def parse_binary(text: str) -> float | None:
    t = text.strip().upper()
    # strict: starts with YES/NO
    if t.startswith("YES"): return 1.0
    if t.startswith("NO"):  return 0.0
    # lenient: word boundary anywhere
    if re.search(r'\bYES\b', t): return 1.0
    if re.search(r'\bNO\b',  t): return 0.0
    return None

def parse_dollar_0_3(text: str) -> float | None:
    m = re.search(r"[-+]?\d+\.?\d*", text.strip())
    return max(0.0, min(3.0, float(m.group()))) if m else None

def parse_percent(text: str) -> float | None:
    # prefer an explicit "N%" pattern first, then fall back to first number
    m = re.search(r"(\d+\.?\d*)\s*%", text)
    if not m:
        m = re.search(r"[-+]?\d+\.?\d*", text.strip())
    return max(0.0, min(100.0, float(m.group(1) if m and m.lastindex else m.group()))) if m else None

def parse_job_choice(text: str) -> dict | None:
    # Returns {"job_pref": "A"|"B", "value": 0|1}  (1 = chose the listed-first job)
    t = text.strip().upper()
    if "JOB A" in t or t.startswith("A"):
        return {"job_pref": "A", "value": 1.0}
    if "JOB B" in t or t.startswith("B"):
        return {"job_pref": "B", "value": 0.0}
    return None

PARSERS = {
    "binary":      parse_binary,
    "dollar_0_3":  parse_dollar_0_3,
    "percent":     parse_percent,
    "job_choice":  parse_job_choice,
}

# ---------------------------------------------------------------------------
# Study configs
# ---------------------------------------------------------------------------

# ── seq=46  Hiring Discrimination ─────────────────────────────────────────
# 5 arms. Outcome: pro-women attitude score.
# Ground truth (Qualtrics sample, discrimination-against-woman group):
#   Control=6.0, Same_Effort=4.2, Same_Suffering=6.2,
#   No_Discrimination=8.7, All_Constant=6.1

STUDY_46 = {
    "seq_id": 46,
    "title":  "Hiring Discrimination",
    "outcomes": [
        {"id": "attitude", "response_format": "percent",
         "question": (
             "\nOn a scale from 0 to 100, how wrong was it for the manager to hire "
             "the man? (0 = not wrong at all, 100 = completely wrong)\n"
             "Reply with a single integer only."
         )}
    ],
    "preamble": (
        "Consider the following hypothetical scenario.\n\n"
        "A job is available in a company. A manager has narrowed down the applicants "
        "to the two most qualified people: one man and one woman."
    ),
    "arms": {
        "Control": (
            "\n\nTaking into account all characteristics, the manager knows that the "
            "woman is slightly more qualified and hiring her would bring slightly "
            "higher profits. After considering everything, the manager hires the man."
        ),
        "Same_Effort": (
            "\n\nTaking into account all characteristics, the manager knows that the "
            "woman is slightly more qualified. The manager hires the man.\n\n"
            "Note: The man and the woman have worked equally hard in their career. "
            "Both regularly studied on weekends while their friends were out partying."
        ),
        "Same_Suffering": (
            "\n\nTaking into account all characteristics, the manager knows that the "
            "woman is slightly more qualified. The manager hires the man.\n\n"
            "Note: The man and the woman would suffer equally from not getting the "
            "job. Both are currently unemployed with 4 weeks of savings, both would "
            "find it equally hard to find another job, and neither supports a family."
        ),
        "No_Discrimination": (
            "\n\nTaking into account all characteristics, the manager knows that the "
            "woman is slightly more qualified. The manager hires the man.\n\n"
            "Note: The job is in an industry where there is no gender discrimination. "
            "Studies have shown that neither men nor women face discrimination in "
            "hiring or treatment by coworkers or supervisors in this industry."
        ),
        "All_Constant": (
            "\n\nTaking into account all characteristics, the manager knows that the "
            "woman is slightly more qualified. The manager hires the man.\n\n"
            "Note: The man and the woman would suffer equally from not getting the "
            "job, have worked equally hard in their careers, and the job is in an "
            "industry with no gender discrimination."
        ),
    },
}

# ── seq=29  Legitimacy & Social Desirability (Minarets, dictator game) ────
# 4 arms. Outcome: donation out of $3.
# Ground truth: arm1≈1.02, arm2≈0.69, arm3≈0.92, arm4≈0.85

STUDY_29 = {
    "seq_id": 29,
    "title":  "Legitimacy and Social Desirability (Minarets)",
    "outcomes": [
        {"id": "donation", "response_format": "dollar_0_3",
         "question": (
             "\nHow much of your $3 do you give to this participant? "
             "Reply with a single number between 0 and 3 only (e.g. 1.50)."
         )}
    ],
    "preamble": (
        "You are taking part in a study about economic decision-making.\n\n"
        "You have been given $3. You are matched with another participant. "
        "You may give them some or all of your $3, or keep it all. "
        "Your decision is final.\n\n"
        "Here is some information about the participant you are matched with:\n"
    ),
    "arms": {
        "arm1": "The participant is a 24-year-old male from Switzerland.",
        "arm2": (
            "The participant is a 24-year-old male from Switzerland. "
            "He supports the prohibition of the building of minarets in Switzerland."
        ),
        "arm3": (
            "The participant is a 24-year-old male from Switzerland. "
            "Like 57.5% of Swiss respondents, he supports the prohibition of the "
            "building of minarets in Switzerland."
        ),
        "arm4": (
            "The participant is a 24-year-old male from Switzerland. "
            "Building minarets is illegal in Switzerland, following a 2009 referendum. "
            "Like 57.5% of Swiss respondents, he supports this prohibition. "
            "However, he did not vote in the referendum himself."
        ),
    },
}

# ── seq=6  Morality-Efficiency (Kidney Systems) ───────────────────────────
# 3 systems × 5 efficiency levels = 15 arms. Outcome: binary support.
# Ground truth (proportion supporting):
#   public_agency: 0.66/0.79/0.85/0.88/0.88 at 0/4k/9k/14k/19k
#   private_old:   0.47/0.56/0.68/0.69/0.73
#   private_new:   0.36/0.42/0.56/0.64/0.73

_KIDNEY_SYSTEMS = {
    "public_agency": (
        "Under this system, kidney donors can receive $20,000 from a public government "
        "agency. Kidneys are allocated by a government-coordinated agency according to "
        "priority rules based on medical urgency, blood match, time on waiting list, "
        "age, and distance. Donors may opt out of payment and make directed donations."
    ),
    "private_old": (
        "Under this system, kidney donors can receive $20,000 directly from the kidney "
        "recipient (e.g., out of pocket or through private health insurance). A "
        "government agency maintains a registry; transactions occur directly between "
        "recipient and donor or through a private agency. Donors may opt out of payment."
    ),
    "private_new": (
        "Under this system, kidney donors can receive $100,000 worth of non-cash "
        "benefits (such as lifetime health insurance or tax credits) arranged by the "
        "recipient's insurer. A government agency maintains a registry; transactions "
        "are arranged through an authorized intermediary. Donors may opt out."
    ),
}

_EFFICIENCY = {
    "out1": (0,     "no increase in the annual number of kidney transplants"),
    "out2": (4000,  "approximately 4,000 more kidney transplants per year"),
    "out3": (9000,  "approximately 9,000 more kidney transplants per year"),
    "out4": (14000, "approximately 14,000 more kidney transplants per year"),
    "out5": (19000, "approximately 19,000 more kidney transplants per year"),
}

_ARMS_6 = {}
for sys_id, sys_text in _KIDNEY_SYSTEMS.items():
    for out_id, (n, label) in _EFFICIENCY.items():
        _ARMS_6[f"{sys_id}__{out_id}"] = (
            sys_text + f"\n\nExpected impact: This system would result in {label}."
        )

STUDY_6 = {
    "seq_id": 6,
    "title":  "Morality-Efficiency Tradeoffs (Kidney Systems)",
    "outcomes": [
        {"id": "support", "response_format": "binary",
         "question": "\nDo you support this system? Reply with YES or NO only."}
    ],
    "preamble": (
        "Currently in the United States, kidney donors receive no compensation. "
        "Each year, approximately 35,000 Americans need a kidney transplant, but "
        "only around 17,000 kidneys are procured.\n\n"
        "Below is a proposed alternative system:\n"
    ),
    "arms": _ARMS_6,
}

# ── seq=28  Islamophobia, Legitimacy & Social Desirability ────────────────
# 6 arms = 3 info conditions × 2 privacy conditions.
# Outcomes: (1) belief about % supporting ban, (2) binary donation to ACT.
# Ground truth belief: A1=45.9, A2=43.1, A3=49.5, A4=54.9, A5=50.9, A6=44.8
# Ground truth donation: A1=0.40, A2=0.29, A3=0.35, A4=0.37, A5=0.29, A6=0.30

_INFO_28 = {
    "Control": "",
    "Popular": (
        "\nNote: A survey found that a large proportion of Americans support "
        "a policy that would prohibit Muslims from holding public office."
    ),
    "Popular_Unconstitutional": (
        "\nNote: A survey found that a large proportion of Americans support "
        "a policy that would prohibit Muslims from holding public office. "
        "However, legal experts have noted that such a policy would be "
        "unconstitutional under the First Amendment."
    ),
}

_PRIVACY_28 = {
    "Private": (
        "\nYour responses in this survey are completely private "
        "and will not be shared with anyone outside the research team."
    ),
    "Public": (
        "\nNote: A member of the research team may personally contact you "
        "to verify your answers to this survey."
    ),
}

_ARM_MAP_28 = {
    "A1": ("Control",                "Private"),
    "A2": ("Control",                "Public"),
    "A3": ("Popular",                "Private"),
    "A4": ("Popular",                "Public"),
    "A5": ("Popular_Unconstitutional","Private"),
    "A6": ("Popular_Unconstitutional","Public"),
}

_ARMS_28 = {
    arm_id: _INFO_28[info] + _PRIVACY_28[priv]
    for arm_id, (info, priv) in _ARM_MAP_28.items()
}

STUDY_28 = {
    "seq_id": 28,
    "title":  "Islamophobia, Legitimacy & Social Desirability",
    "outcomes": [
        {"id": "belief_pct", "response_format": "percent",
         "question": (
             "\n\nWhat percentage of Americans support a policy prohibiting Muslims "
             "from holding public office? Reply with a single integer only."
         )},
        {"id": "donation", "response_format": "binary",
         "question": (
             "\n\nWould you donate to ACT for America, an organization that campaigns "
             "for stricter policies targeting radical Islam? Reply YES or NO only."
         )},
    ],
    "preamble": (
        "We are studying public attitudes toward policies related to religion "
        "and national security in the United States.\n"
    ),
    "arms": _ARMS_28,
}

# ── seq=20  Xenophobia & Social Desirability ──────────────────────────────
# 8 arms across two experiments.
# Exp 1: Control vs ProbInfo × Private vs Public
# Exp 2: TrumpWin vs ClintonWin × Private vs Public
#
# Ground truth E1 belief: Ctrl_Priv=61.5, Ctrl_Pub=63.7, Treat_Priv=66.8, Treat_Pub=68.1
# Ground truth E2 belief: Clinton_Priv=42.1, Clinton_Pub=42.1, Trump_Priv=49.8, Trump_Pub=48.8

_PRIVACY_20 = {
    "Private": (
        "\nYour responses in this survey are completely private."
    ),
    "Public": (
        "\nNote: A member of the research team may personally contact you "
        "to verify your answers."
    ),
}

_INFO_E1 = {
    "Ctrl":      "",
    "TreatProb": (
        "\nNote: According to political scientists, there is approximately "
        "a 35% chance that a presidential candidate openly expressing xenophobic "
        "views toward immigrants will win the next US presidential election."
    ),
}

_INFO_E2 = {
    "ClintonWin": (
        "\nBackground: In the 2016 US Presidential Election, "
        "Hillary Clinton won Pittsburgh's metropolitan area."
    ),
    "TrumpWin": (
        "\nBackground: In the 2016 US Presidential Election, "
        "Donald Trump won Pittsburgh's metropolitan area."
    ),
}

_ARM_MAP_20 = {
    "E1_Ctrl_Private":       ("E1", "Ctrl",      "Private"),
    "E1_Ctrl_Public":        ("E1", "Ctrl",      "Public"),
    "E1_TreatProb_Private":  ("E1", "TreatProb", "Private"),
    "E1_TreatProb_Public":   ("E1", "TreatProb", "Public"),
    "E2_ClintonWin_Private": ("E2", "ClintonWin","Private"),
    "E2_ClintonWin_Public":  ("E2", "ClintonWin","Public"),
    "E2_TrumpWin_Private":   ("E2", "TrumpWin",  "Private"),
    "E2_TrumpWin_Public":    ("E2", "TrumpWin",  "Public"),
}

_ARMS_20 = {}
for arm_id, (exp, info, priv) in _ARM_MAP_20.items():
    if exp == "E1":
        _ARMS_20[arm_id] = _INFO_E1[info] + _PRIVACY_20[priv]
    else:
        _ARMS_20[arm_id] = _INFO_E2[info] + _PRIVACY_20[priv]

_PREAMBLE_E1 = (
    "We are conducting a survey about immigration and social attitudes in the US."
)
_PREAMBLE_E2 = (
    "We are conducting a survey about political attitudes in Pittsburgh, PA."
)

def _preamble_20(arm_id):
    return _PREAMBLE_E1 if arm_id.startswith("E1") else _PREAMBLE_E2

def _question_20(arm_id):
    if arm_id.startswith("E1"):
        return (
            "\n\nWhat percentage of Americans hold xenophobic views toward immigrants? "
            "Reply with a single integer only."
        )
    else:
        return (
            "\n\nWhat percentage of Pittsburgh voters would endorse stricter "
            "immigration enforcement? Reply with a single integer only."
        )

# seq=20 needs custom prompt builder (preamble varies by arm)
STUDY_20 = {
    "seq_id": 20,
    "title":  "Xenophobia and Social Desirability",
    "outcomes": [
        {"id": "belief_pct", "response_format": "percent",
         "question": None},   # question set dynamically per arm below
    ],
    "preamble": None,          # set dynamically per arm
    "arms": _ARMS_20,
    "dynamic_prompt": True,    # flag for custom builder
}

# ── seq=52  Pay-for-Performance Conjoint (Community Demographics) ─────────
# Fixed all attributes except Community demographics.
# Arms: mostly_african_american, mostly_hispanic, mostly_white, multiracial
# Outcome: binary job choice (proportion preferring the job with that demographics)
# Ground truth: AA=0.488, Hisp=0.470, White=0.475, Multi=0.588

_JOB_BASE = (
    "Job title: Community Active Worker\n"
    "Program: Project Hope (city government community empowerment program)\n"
    "Total pay: About average compared to similar jobs\n"
    "Performance bonuses: Fixed salary (no performance bonus)\n"
    "Job performance evaluation: A supervisor evaluation of your work\n"
    "Current community involvement: Moderate participation\n"
    "Community income: Average income neighborhood\n"
    "Overtime work: Occasionally required\n"
    "Key job task: Direct interaction with community residents\n"
    "Community demographics: {demo}"
)

_DEMOS_52 = {
    "mostly_african_american": "Mostly African American",
    "mostly_hispanic":         "Mostly Hispanic",
    "mostly_white":            "Mostly white",
    "multiracial":             "Multiracial",
}

_ARMS_52 = {
    arm_id: _JOB_BASE.format(demo=demo_label)
    for arm_id, demo_label in _DEMOS_52.items()
}

STUDY_52 = {
    "seq_id": 52,
    "title":  "Pay-for-Performance Conjoint (Community Demographics)",
    "outcomes": [
        {"id": "job_attraction", "response_format": "binary",
         "question": "\n\nWould you choose this job? Reply YES or NO only."}
    ],
    "preamble": (
        "You are evaluating a job opportunity at a city government. "
        "Read the job profile below carefully.\n\n"
    ),
    "arms": _ARMS_52,
}

# ── seq=53  Job Characteristics Conjoint (Performance Bonus) ─────────────
# Vary Performance bonuses only; all other attributes fixed.
# Main treatment: bonus (any level) vs no bonus.
# Note: PSM/efficacy subgroup analysis not replicated (requires pre-measured traits).
# Mapping: no_bonus → average of psm_*_no_bonus arms; bonus → average of psm_*_bonus arms
# Ground truth (PSM subgroup, no_bonus≈0.479, bonus≈0.507)

_BONUS_LEVELS = {
    "no_bonus":    "Fixed salary (no performance bonus)",
    "bonus_small": "A small part of your potential pay (5%) is performance bonus",
    "bonus_mod":   "A moderate part of your potential pay (10%) is performance bonus",
    "bonus_large": "A large part of your potential pay (20%) is performance bonus",
}

_ARMS_53 = {
    arm_id: _JOB_BASE.format(demo="Multiracial").replace(
        "Fixed salary (no performance bonus)", bonus_text
    )
    for arm_id, bonus_text in _BONUS_LEVELS.items()
}

STUDY_53 = {
    "seq_id": 53,
    "title":  "Job Characteristics Conjoint (Performance Bonus)",
    "outcomes": [
        {"id": "job_attraction", "response_format": "binary",
         "question": "\n\nWould you choose this job? Reply YES or NO only."}
    ],
    "preamble": (
        "You are evaluating a job opportunity at a city government. "
        "Read the job profile below carefully.\n\n"
    ),
    "arms": _ARMS_53,
}

# ---------------------------------------------------------------------------
# All studies
# ---------------------------------------------------------------------------

STUDY_CONFIGS = {
    6:  STUDY_6,
    20: STUDY_20,
    28: STUDY_28,
    29: STUDY_29,
    46: STUDY_46,
    52: STUDY_52,
    53: STUDY_53,
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(config: dict, arm_id: str, outcome: dict) -> str:
    if config.get("dynamic_prompt") and config["seq_id"] == 20:
        preamble  = _preamble_20(arm_id)
        arm_text  = config["arms"][arm_id]
        question  = _question_20(arm_id)
    else:
        preamble  = config["preamble"]
        arm_text  = config["arms"][arm_id]
        question  = outcome["question"]
    return preamble + arm_text + question

# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

async def simulate_one(client: AsyncOpenAI, seq_id: int, arm_id: str,
                        outcome: dict, prompt: str) -> dict:
    pid = str(uuid.uuid4())
    try:
        r = await client.chat.completions.create(
            model=MODEL,
            temperature=1,
            reasoning_effort="none",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        text   = (r.choices[0].message.content or "").strip()
        parser = PARSERS[outcome["response_format"]]
        raw    = parser(text)
        value  = raw["value"] if isinstance(raw, dict) else raw
        return {
            "seq_id":      seq_id,
            "arm_id":      arm_id,
            "outcome_id":  outcome["id"],
            "pid":         pid,
            "response":    text,
            "value":       value,
            "parse_ok":    value is not None,
        }
    except Exception as e:
        return {
            "seq_id":      seq_id,
            "arm_id":      arm_id,
            "outcome_id":  outcome["id"],
            "pid":         pid,
            "response":    None,
            "value":       None,
            "parse_ok":    False,
            "error":       str(e),
        }


async def run_study(client, config, n_per_arm, writer, pbar):
    seq_id   = config["seq_id"]
    outcomes = config["outcomes"]
    tasks    = []
    for arm_id in config["arms"]:
        for outcome in outcomes:
            prompt = build_prompt(config, arm_id, outcome)
            for _ in range(n_per_arm):
                tasks.append(simulate_one(client, seq_id, arm_id, outcome, prompt))

    for coro in asyncio.as_completed(tasks):
        rec = await coro
        writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pbar.update(1)


def print_summary(records: list[dict]):
    sums = defaultdict(list)
    for r in records:
        if r["parse_ok"] and r["value"] is not None:
            sums[(r["seq_id"], r["outcome_id"], r["arm_id"])].append(r["value"])
    failures = sum(1 for r in records if not r["parse_ok"])
    print(f"\n── Simulated arm means ({len(records)} records, {failures} parse failures) ──")
    for (seq_id, out_id, arm_id), vals in sorted(sums.items()):
        print(f"  seq={seq_id} {out_id:<15} {arm_id:<45} "
              f"mean={sum(vals)/len(vals):.4f}  n={len(vals)}")


# ---------------------------------------------------------------------------
# Async mode
# ---------------------------------------------------------------------------

async def run_async(n_per_arm: int):
    client = AsyncOpenAI()
    total  = sum(len(c["arms"]) * len(c["outcomes"]) * n_per_arm
                 for c in STUDY_CONFIGS.values())
    print(f"Studies: {list(STUDY_CONFIGS)}  |  n_per_arm={n_per_arm}  |  total_calls={total}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(OUTPUT, "w") as f:
        with tqdm(total=total, unit="call") as pbar:
            for seq_id, config in STUDY_CONFIGS.items():
                pbar.set_description(f"seq={seq_id}")
                await run_study(client, config, n_per_arm, f, pbar)

    with open(OUTPUT) as f:
        records = [json.loads(l) for l in f]

    print_summary(records)
    print(f"Output → {OUTPUT}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def build_batch_requests(n_per_arm: int) -> list[dict]:
    requests = []
    for seq_id, config in STUDY_CONFIGS.items():
        for arm_id in config["arms"]:
            for outcome in config["outcomes"]:
                prompt = build_prompt(config, arm_id, outcome)
                for i in range(n_per_arm):
                    requests.append({
                        "custom_id": f"{seq_id}__{arm_id}__{outcome['id']}__{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": MODEL,
                            "temperature": 1,
                            "reasoning_effort": "none",
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user",   "content": prompt},
                            ],
                        },
                    })
    return requests


def submit_batch(n_per_arm: int) -> str:
    client   = OpenAI()
    requests = build_batch_requests(n_per_arm)
    total    = len(requests)
    print(f"Submitting batch: {total} requests …")

    content = "\n".join(json.dumps(r) for r in requests).encode()
    file_obj = client.files.create(file=io.BytesIO(content), purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch submitted. ID: {batch.id}  status: {batch.status}")
    print(f"Run with --download {batch.id} once complete.")
    return batch.id


def download_batch(batch_id: str):
    client = OpenAI()
    print(f"Checking batch {batch_id} …")
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  status={batch.status}  "
              f"completed={counts.completed}/{counts.total}  failed={counts.failed}")
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status: {batch.status}")
        time.sleep(30)

    raw = client.files.content(batch.output_file_id).text
    records = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # custom_id = "{seq_id}__{arm_id}__{out_id}__{i}"
        # arm_id may contain "__" (e.g. private_new__out1), so parse from both ends
        parts  = r["custom_id"].split("__")
        seq_id = int(parts[0])
        out_id = parts[-2]               # second-to-last is always outcome_id
        arm_id = "__".join(parts[1:-2])  # everything between is arm_id
        config  = STUDY_CONFIGS[seq_id]
        outcome = next(o for o in config["outcomes"] if o["id"] == out_id)

        if r.get("error"):
            records.append({"seq_id": seq_id, "arm_id": arm_id, "outcome_id": out_id,
                             "pid": r["custom_id"], "response": None,
                             "value": None, "parse_ok": False, "error": str(r["error"])})
        else:
            text   = (r["response"]["body"]["choices"][0]["message"]["content"] or "").strip()
            parser = PARSERS[outcome["response_format"]]
            raw_v  = parser(text)
            value  = raw_v["value"] if isinstance(raw_v, dict) else raw_v
            records.append({"seq_id": seq_id, "arm_id": arm_id, "outcome_id": out_id,
                             "pid": r["custom_id"], "response": text,
                             "value": value, "parse_ok": value is not None})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print_summary(records)
    print(f"Output → {OUTPUT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int, default=50, dest="n_per_arm")
    parser.add_argument("--mode",     choices=["async", "batch"], default="async")
    parser.add_argument("--download", metavar="BATCH_ID", default=None,
                        help="Download and parse a completed batch job")
    args = parser.parse_args()

    if args.download:
        download_batch(args.download)
    elif args.mode == "batch":
        submit_batch(args.n_per_arm)
    else:
        asyncio.run(run_async(args.n_per_arm))
