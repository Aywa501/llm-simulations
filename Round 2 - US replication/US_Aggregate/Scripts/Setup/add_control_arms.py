"""
add_control_arms.py

One-time script that stamps control_arm_id onto each study's instrument
block in simulatable_studies.jsonl.

control_arm_id is the arm used as the baseline when computing treatment
effects (Δ = treatment_mean − control_mean).  Studies whose design has no
natural single baseline (factorial / all-treatment designs) receive None;
extract_effects_gt.py will skip those.

Run once before extract_effects_gt.py.
"""

import json
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
DATA_DIR     = SCRIPT_DIR.parents[1] / "Data"
STUDIES_PATH = DATA_DIR / "Ground_Truth" / "simulatable_studies.jsonl"

# ---------------------------------------------------------------------------
# Control-arm mapping
#
#   seq=19  — "current_system" is the status-quo baseline; all other arms
#             are alternative organ-procurement systems under evaluation.
#   seq=103 — explicit two-arm RCT.
#   seq=118 — three different framing conditions, no natural baseline → None.
#   seq=125 — 2×2×2 factorial (h/l combinations) → None.
#   seq=131 — three charity-type arms, no control → None.
#   seq=141 — explicit control_group.
#   seq=145 — factorial loan-repayment parameters → None.
#   seq=150 — four-arm RCT with explicit control.
#   seq=151 — explicit controlgroup (flagged unsound but mapped for completeness).
#   seq=152 — discrimination-vignette design, no single baseline → None.
#   seq=159 — two sub-experiments.  "control_group" is the primary baseline
#             (belief-update outcomes); "control" is the alias for the second
#             sub-experiment (budget-allocation outcome).  Both share the
#             same no-score-revelation condition.  Primary stored here;
#             extract_effects_gt.py treats both as the same condition.
#   seq=162 — 36-arm job-characteristics factorial → None.
#   seq=164 — explicit control (flagged unsound but mapped for completeness).
#   seq=169 — giving/redistribution game, no baseline → None.
#   seq=174 — nine different COVID-information messages, no control → None.
#   seq=176 — complex: three ideology groups × control/treatment → None.
#   seq=178 — "nigroup" = no-instruction group, explicit control.
# ---------------------------------------------------------------------------
CONTROL_ARMS: dict[int, str | None] = {
    19:  "current_system",
    103: "control",
    118: None,
    125: None,
    131: None,
    141: "control_group",
    145: None,
    150: "control",
    151: "controlgroup",
    152: None,
    159: "control_group",
    162: None,
    164: "control",
    169: None,
    174: None,
    176: None,
    178: "nigroup",
}


def main():
    records = []
    with open(STUDIES_PATH) as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["seq_id"]
            ctrl = CONTROL_ARMS.get(sid)
            rec.setdefault("instrument", {})["control_arm_id"] = ctrl
            records.append(rec)

    with open(STUDIES_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    has_ctrl  = sum(1 for r in records
                    if r["instrument"].get("control_arm_id") is not None)
    print(f"Updated {len(records)} studies.")
    print(f"  {has_ctrl} with a control arm  →  eligible for effects extraction")
    print(f"  {len(records) - has_ctrl} without a control arm  →  skipped")
    print()
    for rec in records:
        ctrl = rec["instrument"].get("control_arm_id")
        print(f"  seq={rec['seq_id']:>3}  control_arm_id={ctrl}")


if __name__ == "__main__":
    main()
