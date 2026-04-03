"""
simulate_experiments.py

Participant-mode LLM simulation for 7 studies (6 green + seq=46 yellow).
Outputs per batch config: Data/Microdata/simulation_raw_{cfg}.jsonl

Note: seq=44 (Expectation Formation) excluded — ground truth accuracy scores
were computed from the actual experimental time series, which is not available.
Synthetic data would produce incomparable accuracy metrics.

Usage:
    python simulate_experiments.py --mode batch --config no_reasoning [--n 100]
    python simulate_experiments.py --mode batch --config reasoning_low [--n 100]
    python simulate_experiments.py --mode batch --config reasoning_medium [--n 100]
    python simulate_experiments.py --all-batches [--n 100]   # submits all 3

    python simulate_experiments.py --download BATCH_ID --config <cfg>
    python simulate_experiments.py --mode async --config no_reasoning [--n 100]
"""

import argparse, asyncio, io, json, re, time, uuid
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from openai import AsyncOpenAI, OpenAI

DATA_DIR = Path(__file__).resolve().parents[2] / "Data" / "Microdata"
MODEL    = "gpt-5.1"

# ---------------------------------------------------------------------------
# Batch configurations
# ---------------------------------------------------------------------------

BATCH_CONFIGS = {
    "no_reasoning": {
        "label":        "Batch 1 — no reasoning",
        "model_params": {"temperature": 1, "top_p": 1},
    },
    "reasoning_low": {
        "label":        "Batch 2 — reasoning low",
        "model_params": {"reasoning_effort": "low"},
    },
    "reasoning_medium": {
        "label":        "Batch 3 — reasoning medium",
        "model_params": {"reasoning_effort": "medium"},
    },
}

def output_path(cfg_name: str) -> Path:
    return DATA_DIR / f"simulation_raw_{cfg_name}.jsonl"

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

def parse_choice_ab(text: str) -> float | None:
    """Forced A/B choice. Returns 1.0 if A chosen (target job), 0.0 if B."""
    t = text.strip().upper()
    if t.startswith("A") or "JOB A" in t: return 1.0
    if t.startswith("B") or "JOB B" in t: return 0.0
    if re.search(r'\bA\b', t): return 1.0
    if re.search(r'\bB\b', t): return 0.0
    return None

PARSERS = {
    "binary":      parse_binary,
    "dollar_0_3":  parse_dollar_0_3,
    "percent":     parse_percent,
    "choice_ab":   parse_choice_ab,
}

# ---------------------------------------------------------------------------
# Study configs
# ---------------------------------------------------------------------------

# ── seq=46  Hiring Discrimination ─────────────────────────────────────────
# 4 arms (2×2): presentation order × direction of discrimination.
# Matches ground truth arm labels exactly.
# Ground truth (microdata "judgment" variable — NOTE: different scale from 0–100,
#   values are ~1–13; comparison uses rank/direction not absolute values):
#   women_first_woman_discrimination = 13.26  (woman presented first, woman discriminated)
#   woman_first_man_discrimination   =  9.00  (woman presented first, man discriminated)
#   man_first_woman_discrimination   =  8.01  (man presented first, woman discriminated)
#   man_first_man_discrimination     =  1.48  (man presented first, man discriminated)

_HIRING_QUESTION = (
    "\n\nOn a scale from 0 to 100, how wrong was it for the manager to hire "
    "the less qualified candidate? (0 = not wrong at all, 100 = completely wrong)\n"
    "Reply with a single integer only."
)

STUDY_46 = {
    "seq_id": 46,
    "title":  "Hiring Discrimination (2×2 order × direction)",
    "outcomes": [
        {"id": "judgment", "response_format": "percent",
         "question": _HIRING_QUESTION}
    ],
    "preamble": (
        "Consider the following hypothetical scenario.\n\n"
        "A job is available in a company. A manager has narrowed down the applicants "
        "to the two most qualified people."
    ),
    "arms": {
        # Woman described first, woman discriminated against (manager hires man)
        "women_first_woman_discrimination": (
            " The two candidates are Sarah (a woman) and James (a man).\n\n"
            "Taking into account all characteristics, the manager knows that Sarah "
            "is slightly more qualified and hiring her would bring slightly higher "
            "profits. After considering everything, the manager hires James."
        ),
        # Woman described first, man discriminated against (manager hires woman)
        "woman_first_man_discrimination": (
            " The two candidates are Sarah (a woman) and James (a man).\n\n"
            "Taking into account all characteristics, the manager knows that James "
            "is slightly more qualified and hiring him would bring slightly higher "
            "profits. After considering everything, the manager hires Sarah."
        ),
        # Man described first, woman discriminated against (manager hires man)
        "man_first_woman_discrimination": (
            " The two candidates are James (a man) and Sarah (a woman).\n\n"
            "Taking into account all characteristics, the manager knows that Sarah "
            "is slightly more qualified and hiring her would bring slightly higher "
            "profits. After considering everything, the manager hires James."
        ),
        # Man described first, man discriminated against (manager hires woman)
        "man_first_man_discrimination": (
            " The two candidates are James (a man) and Sarah (a woman).\n\n"
            "Taking into account all characteristics, the manager knows that James "
            "is slightly more qualified and hiring him would bring slightly higher "
            "profits. After considering everything, the manager hires Sarah."
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

# ── seq=53  Job Characteristics Conjoint (Performance Bonus) — forced choice
# Two arms: target job has bonus vs target job has no bonus.
# Response: A (chose target) = 1.0, B (chose alternative) = 0.0.
# Ground truth (PSM subgroup, weighted across high/low PSM):
#   has_bonus  ≈ 0.509  (psm_high_bonus=0.507 n=3450, psm_low_bonus=0.510 n=3241)
#   no_bonus   ≈ 0.476  (psm_high_no_bonus=0.479 n=1164, psm_low_no_bonus=0.473 n=1151)

_JOB_53_FIXED = (
    "Job title: Community Active Worker\n"
    "Program: Project Hope (city government community empowerment program)\n"
    "Total pay: About average compared to similar jobs\n"
    "Job performance evaluation: A supervisor evaluation of your work\n"
    "Community involvement: Moderate participation\n"
    "Community income: Average income neighborhood\n"
    "Community demographics: Multiracial\n"
    "Overtime work: Occasionally required\n"
    "Key job task: Direct interaction with community residents\n"
    "Performance bonuses: {bonus}"
)

_BONUS_YES = "A moderate part of your potential pay (10%) is performance bonus"
_BONUS_NO  = "Fixed salary (no performance bonus)"

STUDY_53 = {
    "seq_id": 53,
    "title":  "Job Characteristics Conjoint (Performance Bonus) — forced choice",
    "outcomes": [
        {"id": "job_attraction", "response_format": "choice_ab",
         "question": "\n\nWhich job would you prefer? Reply A or B only."}
    ],
    "preamble": (
        "You are choosing between two job opportunities at a city government. "
        "The jobs are identical except where noted below.\n\n"
    ),
    "arms": {
        # Target (Job A) has bonus; alternative (Job B) has no bonus
        "has_bonus": (
            "Job A:\n" + _JOB_53_FIXED.format(bonus=_BONUS_YES) + "\n\n"
            "Job B:\n" + _JOB_53_FIXED.format(bonus=_BONUS_NO)
        ),
        # Target (Job A) has no bonus; alternative (Job B) has bonus
        "no_bonus": (
            "Job A:\n" + _JOB_53_FIXED.format(bonus=_BONUS_NO) + "\n\n"
            "Job B:\n" + _JOB_53_FIXED.format(bonus=_BONUS_YES)
        ),
    },
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
                        outcome: dict, prompt: str, model_params: dict) -> dict:
    pid = str(uuid.uuid4())
    try:
        r = await client.chat.completions.create(
            model=MODEL,
            **model_params,
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


async def run_study(client, config, n_per_arm, writer, pbar, model_params):
    seq_id   = config["seq_id"]
    outcomes = config["outcomes"]
    tasks    = []
    for arm_id in config["arms"]:
        for outcome in outcomes:
            prompt = build_prompt(config, arm_id, outcome)
            for _ in range(n_per_arm):
                tasks.append(simulate_one(client, seq_id, arm_id, outcome, prompt, model_params))

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

async def run_async(n_per_arm: int, cfg_name: str):
    cfg          = BATCH_CONFIGS[cfg_name]
    model_params = cfg["model_params"]
    out          = output_path(cfg_name)
    client       = AsyncOpenAI()
    total        = sum(len(c["arms"]) * len(c["outcomes"]) * n_per_arm
                       for c in STUDY_CONFIGS.values())
    print(f"{cfg['label']}  |  studies={list(STUDY_CONFIGS)}  |  n={n_per_arm}  |  calls={total}")
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        with tqdm(total=total, unit="call") as pbar:
            for seq_id, config in STUDY_CONFIGS.items():
                pbar.set_description(f"seq={seq_id}")
                await run_study(client, config, n_per_arm, f, pbar, model_params)

    with open(out) as f:
        records = [json.loads(l) for l in f]

    print_summary(records)
    print(f"Output → {out}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def build_batch_requests(n_per_arm: int, model_params: dict) -> list[dict]:
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
                            **model_params,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user",   "content": prompt},
                            ],
                        },
                    })
    return requests


def submit_batch(n_per_arm: int, cfg_name: str) -> str:
    cfg          = BATCH_CONFIGS[cfg_name]
    model_params = cfg["model_params"]
    client       = OpenAI()
    requests     = build_batch_requests(n_per_arm, model_params)
    print(f"{cfg['label']}  |  {len(requests)} requests …")

    content  = "\n".join(json.dumps(r) for r in requests).encode()
    file_obj = client.files.create(file=io.BytesIO(content), purpose="batch")
    batch    = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Submitted  cfg={cfg_name}  batch_id={batch.id}  status={batch.status}")
    print(f"Download:  python simulate_experiments.py --download {batch.id} --config {cfg_name}")
    return batch.id


def download_batch(batch_id: str, cfg_name: str):
    client = OpenAI()
    out    = output_path(cfg_name)
    print(f"Checking batch {batch_id}  cfg={cfg_name} …")
    while True:
        batch  = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  status={batch.status}  completed={counts.completed}/{counts.total}  failed={counts.failed}")
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status: {batch.status}")
        time.sleep(30)

    # If everything failed, output_file_id is None — read error file instead to diagnose
    if batch.output_file_id is None:
        if batch.error_file_id:
            err_raw = client.files.content(batch.error_file_id).text
            sample  = err_raw.splitlines()[:3]
            for s in sample:
                print("  ERROR SAMPLE:", s[:300])
        raise RuntimeError(f"Batch {batch_id}: 0 successes, {counts.failed} failures. See error samples above.")

    raw     = client.files.content(batch.output_file_id).text
    records = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        r      = json.loads(line)
        parts  = r["custom_id"].split("__")
        seq_id = int(parts[0])
        out_id = parts[-2]
        arm_id = "__".join(parts[1:-2])
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
                             "value": value, "parse_ok": value is not None,
                             "batch_cfg": cfg_name})

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print_summary(records)
    print(f"Output → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=100, dest="n_per_arm")
    parser.add_argument("--mode",        choices=["async", "batch"], default="batch")
    parser.add_argument("--config",      choices=list(BATCH_CONFIGS), default="no_reasoning",
                        help="Which batch config to use")
    parser.add_argument("--all-batches", action="store_true",
                        help="Submit all 3 batch configs in sequence")
    parser.add_argument("--download",    metavar="BATCH_ID", default=None,
                        help="Download and parse a completed batch job (requires --config)")
    args = parser.parse_args()

    if args.download:
        download_batch(args.download, args.config)
    elif args.all_batches:
        for cfg_name in BATCH_CONFIGS:
            submit_batch(args.n_per_arm, cfg_name)
    elif args.mode == "batch":
        submit_batch(args.n_per_arm, args.config)
    else:
        asyncio.run(run_async(args.n_per_arm, args.config))
