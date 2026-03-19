import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "study_enriched_requested.jsonl"

SACKEY_OUT = ROOT / "rct_public_data_sankey.png"
ARM_HIST_OUT = ROOT / "participants_by_arm_histogram.png"
STUDY_HIST_OUT = ROOT / "participants_by_study_histogram.png"

TOTAL_PUBLIC = 311
AMERICAN = 57
NON_AMERICAN = TOTAL_PUBLIC - AMERICAN

CATEGORY_MAP = {
    "local_microdata_available_now": [8, 13, 15, 19, 23, 24, 26, 28, 29, 34, 36, 38, 41, 42, 43, 44, 45, 46, 47, 48, 50, 52, 53, 54, 55, 57],
    "local_archive_likely_contains_microdata": [1, 2, 3, 6, 9, 10, 12, 14, 20, 51],
    "local_aggregate_or_paper_extraction": [16, 21, 22],
    "local_code_docs_questionnaires_only": [7, 11, 17, 27, 33, 35, 49],
    "no_meaningful_material": [4, 5, 18, 25, 30, 31, 32, 37, 39, 40, 56],
}

TIER_MAP = {
    "tier1": [6, 20, 22, 28, 29, 38, 41, 42, 44, 45, 46, 50, 51, 52, 53, 57],
    "tier2": [2, 10, 19, 23, 24, 47, 48, 55],
}


def load_records():
    records = {}
    for line in DATASET_PATH.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            records[obj["seq_id"]] = obj
    return records


def seq_to_category():
    out = {}
    for category, seqs in CATEGORY_MAP.items():
        for seq in seqs:
            out[seq] = category
    return out


def seq_to_tier():
    tier_lookup = {}
    for tier, seqs in TIER_MAP.items():
        for seq in seqs:
            tier_lookup[seq] = tier
    for seq in range(1, 58):
        tier_lookup.setdefault(seq, "tier3")
    return tier_lookup


def human_label(key):
    return {
        "all_public": "All RCTs with\npublic data links\nn=311",
        "american": "American data\nn=57",
        "non_american": "Other / non-US\nn=254",
        "local_microdata_available_now": "Local microdata\navailable now\nn=26",
        "local_archive_likely_contains_microdata": "Local archive likely\ncontains microdata\nn=10",
        "local_aggregate_or_paper_extraction": "Aggregate / derived /\npaper-level only\nn=3",
        "local_code_docs_questionnaires_only": "Code / docs /\nquestionnaires only\nn=7",
        "no_meaningful_material": "No meaningful\nmaterial\nn=11",
        "tier1": "Tier 1\nn=16",
        "tier2": "Tier 2\nn=8",
        "tier3": "Tier 3\nn=33",
    }[key]


NODE_COLORS = {
    "all_public": "#2A4B7C",
    "american": "#4C956C",
    "non_american": "#B8C0CC",
    "local_microdata_available_now": "#4C956C",
    "local_archive_likely_contains_microdata": "#8FB339",
    "local_aggregate_or_paper_extraction": "#F4A259",
    "local_code_docs_questionnaires_only": "#D97D54",
    "no_meaningful_material": "#9AA5B1",
    "tier1": "#2E8B57",
    "tier2": "#E09F3E",
    "tier3": "#B56576",
}


def ribbon(ax, x0, x1, y0_top, y0_bot, y1_top, y1_bot, color, alpha=0.55):
    cx = (x1 - x0) * 0.45
    verts = [
        (x0, y0_top),
        (x0 + cx, y0_top),
        (x1 - cx, y1_top),
        (x1, y1_top),
        (x1, y1_bot),
        (x1 - cx, y1_bot),
        (x0 + cx, y0_bot),
        (x0, y0_bot),
        (x0, y0_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def draw_node(ax, x, y_top, height, label, color, width=0.12):
    ax.add_patch(Rectangle((x - width / 2, y_top - height), width, height, facecolor=color, edgecolor="white", linewidth=1.5))
    ax.text(x, y_top - height / 2, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")


def make_sankey():
    category_counts = {k: len(v) for k, v in CATEGORY_MAP.items()}
    tier_lookup = seq_to_tier()
    flow_counts = defaultdict(Counter)
    for category, seqs in CATEGORY_MAP.items():
        for seq in seqs:
            flow_counts[category][tier_lookup[seq]] += 1

    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    total_height = 0.78
    scale = total_height / TOTAL_PUBLIC
    x_positions = {"col0": 0.10, "col1": 0.33, "col2": 0.60, "col3": 0.86}
    y_top = 0.90

    node_heights = {
        "all_public": TOTAL_PUBLIC * scale,
        "american": AMERICAN * scale,
        "non_american": NON_AMERICAN * scale,
        "tier1": len(TIER_MAP["tier1"]) * scale,
        "tier2": len(TIER_MAP["tier2"]) * scale,
        "tier3": (AMERICAN - len(TIER_MAP["tier1"]) - len(TIER_MAP["tier2"])) * scale,
    }
    node_heights.update({k: v * scale for k, v in category_counts.items()})

    y_nodes = {
        "all_public": y_top,
        "american": y_top,
        "non_american": y_top - node_heights["american"] - 0.04,
    }

    category_order = [
        "local_microdata_available_now",
        "local_archive_likely_contains_microdata",
        "local_aggregate_or_paper_extraction",
        "local_code_docs_questionnaires_only",
        "no_meaningful_material",
    ]
    cur = y_top
    for key in category_order:
        y_nodes[key] = cur
        cur -= node_heights[key] + 0.02

    tier_order = ["tier1", "tier2", "tier3"]
    cur = y_top
    for key in tier_order:
        y_nodes[key] = cur
        cur -= node_heights[key] + 0.06

    draw_node(ax, x_positions["col0"], y_nodes["all_public"], node_heights["all_public"], human_label("all_public"), NODE_COLORS["all_public"])
    draw_node(ax, x_positions["col1"], y_nodes["american"], node_heights["american"], human_label("american"), NODE_COLORS["american"])
    draw_node(ax, x_positions["col1"], y_nodes["non_american"], node_heights["non_american"], human_label("non_american"), NODE_COLORS["non_american"])
    for key in category_order:
        draw_node(ax, x_positions["col2"], y_nodes[key], node_heights[key], human_label(key), NODE_COLORS[key], width=0.16)
    for key in tier_order:
        draw_node(ax, x_positions["col3"], y_nodes[key], node_heights[key], human_label(key), NODE_COLORS[key])

    # Column 0 -> Column 1
    all_cursor = y_nodes["all_public"]
    american_top = y_nodes["american"]
    non_american_top = y_nodes["non_american"]
    ribbon(
        ax,
        x_positions["col0"] + 0.06,
        x_positions["col1"] - 0.06,
        all_cursor,
        all_cursor - node_heights["american"],
        american_top,
        american_top - node_heights["american"],
        NODE_COLORS["american"],
        alpha=0.60,
    )
    ribbon(
        ax,
        x_positions["col0"] + 0.06,
        x_positions["col1"] - 0.06,
        all_cursor - node_heights["american"],
        all_cursor - node_heights["american"] - node_heights["non_american"],
        non_american_top,
        non_american_top - node_heights["non_american"],
        NODE_COLORS["non_american"],
        alpha=0.50,
    )

    # American -> category
    source_cursor = y_nodes["american"]
    category_cursors = {k: y_nodes[k] for k in category_order}
    for key in category_order:
        h = node_heights[key]
        ribbon(
            ax,
            x_positions["col1"] + 0.06,
            x_positions["col2"] - 0.08,
            source_cursor,
            source_cursor - h,
            category_cursors[key],
            category_cursors[key] - h,
            NODE_COLORS[key],
            alpha=0.58,
        )
        source_cursor -= h
        category_cursors[key] -= h

    # Category -> tier
    tier_cursors = {k: y_nodes[k] for k in tier_order}
    for category in category_order:
        cat_cursor = y_nodes[category]
        for tier in tier_order:
            count = flow_counts[category][tier]
            if not count:
                continue
            h = count * scale
            ribbon(
                ax,
                x_positions["col2"] + 0.08,
                x_positions["col3"] - 0.06,
                cat_cursor,
                cat_cursor - h,
                tier_cursors[tier],
                tier_cursors[tier] - h,
                NODE_COLORS[tier],
                alpha=0.50,
            )
            cat_cursor -= h
            tier_cursors[tier] -= h

    ax.set_title("Public-Data RCT Funnel and Simulation Tier Split", fontsize=18, fontweight="bold", pad=18)
    ax.text(0.5, 0.02, "Tier split uses the current Tier 1 / Tier 2 / Tier 3 study classification; code-only and no-material studies are assigned to Tier 3.", ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout()
    fig.savefig(SACKEY_OUT, bbox_inches="tight")
    plt.close(fig)


def preferred_anchor_outcome(results):
    preferred = results.get("preferred_simulation_outcome_ids") or []
    outcomes = results.get("outcomes", [])
    by_id = {o["outcome_id"]: o for o in outcomes}
    for oid in preferred:
        out = by_id.get(oid)
        if out and any(g.get("n_analyzed") is not None for g in out.get("group_summaries", [])):
            return out
    for out in outcomes:
        if any(g.get("n_analyzed") is not None for g in out.get("group_summaries", [])):
            return out
    return None


def make_histograms(records):
    arm_ns = []
    study_ns = []
    for rec in records.values():
        out = preferred_anchor_outcome(rec["enrichment"]["results"])
        if out is None:
            continue
        group_ns = [g["n_analyzed"] for g in out["group_summaries"] if g.get("n_analyzed") is not None]
        if not group_ns:
            continue
        arm_ns.extend(group_ns)
        study_ns.append(sum(group_ns))

    arm_bins = np.logspace(np.log10(min(arm_ns)), np.log10(max(arm_ns)), 16)
    study_bins = np.logspace(np.log10(min(study_ns)), np.log10(max(study_ns)), 16)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    ax.hist(arm_ns, bins=arm_bins, color="#4C956C", edgecolor="white", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_title(f"Participants by Arm\nOne Anchor Outcome per Study, n={len(arm_ns)} Arms", fontsize=16, fontweight="bold")
    ax.set_xlabel("Participants / analyzed observations per arm (log scale)")
    ax.set_ylabel("Number of arms")
    fig.tight_layout()
    fig.savefig(ARM_HIST_OUT, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    ax.hist(study_ns, bins=study_bins, color="#2A4B7C", edgecolor="white", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_title(f"Participants by Study\nOne Anchor Outcome per Study, n={len(study_ns)} Studies", fontsize=16, fontweight="bold")
    ax.set_xlabel("Participants / analyzed observations per study (log scale)")
    ax.set_ylabel("Number of studies")
    fig.tight_layout()
    fig.savefig(STUDY_HIST_OUT, bbox_inches="tight")
    plt.close(fig)


def main():
    records = load_records()
    make_sankey()
    make_histograms(records)
    print(SACKEY_OUT.name)
    print(ARM_HIST_OUT.name)
    print(STUDY_HIST_OUT.name)


if __name__ == "__main__":
    main()
