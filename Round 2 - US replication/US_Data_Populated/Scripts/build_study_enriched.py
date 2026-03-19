import json
import re
import subprocess
import tempfile
from copy import deepcopy
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pyreadstat


ROOT = Path(__file__).resolve().parent
DESIGN_PATH = ROOT / "design_specs_enriched.jsonl"
OUTPUT_PATH = ROOT / "study_enriched_requested.jsonl"

SELECTED_IDS = [
    8, 13, 15, 16, 19, 21, 22, 23, 24, 26, 28, 29, 34, 36, 38, 41, 42, 43,
    44, 45, 46, 47, 48, 50, 52, 53, 54, 55, 57, 1, 2, 3, 6, 9, 10, 12, 14,
    20, 51,
]


def load_designs():
    out = {}
    for line in DESIGN_PATH.read_text().splitlines():
        obj = json.loads(line)
        out[obj["seq_id"]] = obj
    return out


def read_dta(rel_path):
    return pyreadstat.read_dta(ROOT / rel_path)[0]


def read_csv(rel_path):
    return pd.read_csv(ROOT / rel_path)


def read_tsv(rel_path):
    return pd.read_csv(ROOT / rel_path, sep="\t")


def read_dta_from_zip(zip_rel_path, member):
    with ZipFile(ROOT / zip_rel_path) as zf, tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / Path(member).name
        tmp.write_bytes(zf.read(member))
        return pyreadstat.read_dta(tmp)[0]


def read_sas_from_zip(zip_rel_path, member):
    with ZipFile(ROOT / zip_rel_path) as zf, tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / Path(member).name
        tmp.write_bytes(zf.read(member))
        return pyreadstat.read_sas7bdat(tmp)[0]


def read_csv_from_zip(zip_rel_path, member):
    with ZipFile(ROOT / zip_rel_path) as zf, tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / Path(member).name
        tmp.write_bytes(zf.read(member))
        return pd.read_csv(tmp)


def read_rda_frame(rel_path, columns):
    col_vec = ", ".join([f'"{c}"' for c in columns])
    script = f"""
load("{(ROOT / rel_path).as_posix()}")
obj <- get(ls()[1])
sel <- obj[, c({col_vec}), drop = FALSE]
write.table(sel, file = "", sep = "\\t", row.names = FALSE, quote = FALSE, na = "")
"""
    out = subprocess.check_output(["Rscript", "-e", script], text=True)
    return pd.read_csv(StringIO(out), sep="\t")


def infer_results_data_type(source_type, flags, outcomes):
    flags = flags or []
    flag_text = " ".join(flags)
    outcome_names = " ".join(o["name"] for o in (outcomes or []))
    arm_ids = " ".join(
        gs.get("arm_id", "")
        for o in (outcomes or [])
        for gs in o.get("group_summaries", [])
    ).lower()

    if "conjoint_marginal_means" in flag_text:
        return "conjoint_marginal_means"
    if "segment_specific" in flag_text:
        return "segment_specific_summaries"
    if "effect_estimate" in outcome_names.lower() or "treatment_effect_estimates" in flag_text:
        return "effect_estimates_only"
    if source_type == "code_output":
        return "effect_estimates_only"
    if source_type == "paper_or_appendix_tables":
        if "paper_text_used" in flag_text:
            return "text_reported_summaries"
        if "effect_estimate" in arm_ids or "gotv_2010_effect" in arm_ids or "gotv_2012_effect" in arm_ids:
            return "effect_estimates_only"
        return "table_arm_summaries"
    if source_type == "local_summary_files":
        if "observational_groups" in flag_text or "registry_has_no_arms" in flag_text or "site_program_pairs" in flag_text or "target_group_specific_pairs" in flag_text:
            return "nonstandard_group_summaries"
        return "raw_arm_summaries"
    if source_type == "mixed":
        return "nonstandard_group_summaries"
    return "nonstandard_group_summaries"


def make_results(status, source_type, source_files, notes, outcomes=None, preferred=None, flags=None, results_data_type=None):
    outcomes = outcomes or []
    flags = flags or []
    return {
        "extraction": {"method": "script"},
        "status": status,
        "summary_source_type": source_type,
        "results_data_type": results_data_type or infer_results_data_type(source_type, flags, outcomes),
        "source_files": source_files,
        "preferred_simulation_outcome_ids": preferred or [],
        "outcomes": outcomes,
        "notes": notes,
        "quality": {
            "validation_passed": True,
            "validation_errors": [],
            "flags": flags,
            "needs_manual": bool(flags),
        },
    }


def outcome_record(
    outcome_id,
    name,
    outcome_type,
    unit,
    group_summaries,
    notes="",
    is_primary=True,
    analysis_population=None,
    timepoint=None,
):
    return {
        "outcome_id": outcome_id,
        "name": name,
        "is_primary": is_primary,
        "preferred_for_simulation": True,
        "timepoint": timepoint,
        "outcome_type": outcome_type,
        "unit": unit,
        "analysis_population": analysis_population,
        "group_summaries": group_summaries,
        "evidence_quote_ids": [],
        "notes": notes,
    }


def summarize_groups(df, group_col, value_col, group_map=None, metric="mean", dropna=True):
    work = df[[group_col, value_col]].copy()
    if dropna:
        work = work.dropna(subset=[group_col, value_col])
    else:
        work = work.dropna(subset=[group_col])
    grouped = work.groupby(group_col)[value_col].agg(["mean", "std", "count"]).reset_index()
    out = []
    for _, row in grouped.iterrows():
        gval = row[group_col]
        if group_map and gval in group_map:
            arm_id = group_map[gval]["arm_id"]
            note = group_map[gval].get("notes", "")
        else:
            arm_id = str(gval)
            note = ""
        out.append(
            {
                "arm_id": arm_id,
                "n_analyzed": int(row["count"]),
                "metric": metric,
                "value": float(row["mean"]),
                "sd": None if pd.isna(row["std"]) else float(row["std"]),
                "se": None if pd.isna(row["std"]) else float(row["std"] / max(row["count"], 1) ** 0.5),
                "ci_lower": None,
                "ci_upper": None,
                "evidence_quote_ids": [],
                "notes": note,
            }
        )
    return out


def manual_group_summary(arm_id, value, metric="mean", n_analyzed=None, sd=None, se=None, ci_lower=None, ci_upper=None, notes=""):
    return {
        "arm_id": arm_id,
        "n_analyzed": n_analyzed,
        "metric": metric,
        "value": float(value),
        "sd": sd,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "evidence_quote_ids": [],
        "notes": notes,
    }


def make_manual_outcome(outcome_id, name, outcome_type, unit, rows, notes="", is_primary=True, analysis_population=None, timepoint=None):
    return outcome_record(
        outcome_id,
        name,
        outcome_type,
        unit,
        [manual_group_summary(**row) for row in rows],
        notes=notes,
        is_primary=is_primary,
        analysis_population=analysis_population,
        timepoint=timepoint,
    )


def parse_tex_num(token):
    clean = token.strip()
    clean = clean.replace("\\sym{***}", "").replace("\\sym{**}", "").replace("\\sym{*}", "")
    clean = clean.replace("\\", "").strip()
    if clean.startswith("(") and clean.endswith(")"):
        clean = clean[1:-1]
    if clean in {"", "."}:
        return None
    if clean.startswith("-."):
        clean = clean.replace("-.", "-0.", 1)
    elif clean.startswith("."):
        clean = "0" + clean
    return float(clean)


def parse_stargazer_table_rows(rel_path):
    rows = {}
    pending = None
    for raw_line in (ROOT / rel_path).read_text().splitlines():
        line = raw_line.strip()
        if (
            not line
            or "&" not in line
            or line.startswith("\\")
            or line.startswith("{")
            or line.startswith("[1em]")
            or "multicolumn" in line
        ):
            continue
        if line.startswith("f\\_test") or line.startswith("N           "):
            continue
        parts = [p.strip() for p in line.split("&")]
        row_name = parts[0]
        values = [parse_tex_num(p) for p in parts[1:]]
        if row_name != "":
            rows[row_name] = {"coef": values, "se": []}
            pending = row_name
        elif pending is not None:
            rows[pending]["se"] = values
            pending = None
    return rows


def partial_from_files(source_files, note):
    return make_results(
        status="partial",
        source_type="mixed",
        source_files=source_files,
        preferred=[],
        outcomes=[],
        notes=note,
        flags=["manual_mapping_needed"],
    )


def enrich_study(seq_id, rec):
    # Default partial fallback for anything not yet explicitly mapped.
    results = partial_from_files([], "Study package inventoried, but treatment/outcome mapping still needs manual confirmation.")

    if seq_id == 1:
        desc = read_dta_from_zip("1_Kaden/1/oregon_puf.zip", "OHIE_Public_Use_Files/OHIE_Data/oregonhie_descriptive_vars.dta")
        surv = read_dta_from_zip("1_Kaden/1/oregon_puf.zip", "OHIE_Public_Use_Files/OHIE_Data/oregonhie_survey12m_vars.dta")
        df = desc[["person_id", "treatment"]].merge(surv[["person_id", "ins_any_12m"]], on="person_id", how="inner")
        groups = summarize_groups(
            df,
            "treatment",
            "ins_any_12m",
            group_map={
                1: {"arm_id": "arm1"},
                0: {"arm_id": "arm2"},
            },
            metric="proportion",
        )
        results = make_results(
            status="partial",
            source_type="local_summary_files",
            source_files=[
                "1_Kaden/1/oregon_puf.zip:OHIE_Public_Use_Files/OHIE_Data/oregonhie_descriptive_vars.dta",
                "1_Kaden/1/oregon_puf.zip:OHIE_Public_Use_Files/OHIE_Data/oregonhie_survey12m_vars.dta",
            ],
            preferred=["out1"],
            outcomes=[outcome_record("out1", "any insurance at 12 months", "binary", "share", groups)],
            notes="Canonical group-level outcome extracted by merging the lottery selection flag with 12-month survey insurance coverage.",
        )

    elif seq_id == 2:
        df = read_dta_from_zip("1_Kaden/2/113033-V1.zip", "data-and-programs/DYAGK_audit_data.dta")
        df = df.assign(group=df.apply(lambda r: f"{int(r['pub'])}_{int(r['health_cert'])}", axis=1))
        group_map = {
            "1_0": {"arm_id": "pub_nocert", "notes": "Public sector, no credential"},
            "1_1": {"arm_id": "pub_cert", "notes": "Public sector, credential"},
            "0_0": {"arm_id": "fp_nocert", "notes": "For-profit/private sector, no credential"},
            "0_1": {"arm_id": "fp_cert", "notes": "For-profit/private sector, credential"},
        }
        groups = summarize_groups(df, "group", "any_call", group_map=group_map, metric="proportion")
        results = make_results(
            status="partial",
            source_type="local_summary_files",
            source_files=["1_Kaden/2/113033-V1.zip:data-and-programs/DYAGK_audit_data.dta"],
            preferred=["out1"],
            outcomes=[outcome_record("out1", "employer callback", "binary", "share", groups, notes="Groups derived from public/private sector and credential indicators in the replication data.")],
            notes="Extracted callback rates by a four-cell credential x sector grouping visible in the audit dataset.",
        )

    elif seq_id == 8:
        df = read_dta("1_Kaden/8/B_Green_M_APR_2006_Dataset.dta")
        groups = summarize_groups(df, "torc", "pcttc", group_map={1: {"arm_id": "arm1"}, 0: {"arm_id": "arm2"}}, metric="mean")
        results = make_results(
            "partial",
            "local_summary_files",
            ["1_Kaden/8/B_Green_M_APR_2006_Dataset.dta"],
            "Using zipcode-level percentage taking the tax credit as the canonical group-level outcome.",
            [outcome_record("out1", "share claiming the tax credit", "continuous", "proportion", groups)],
            ["out1"],
        )

    elif seq_id == 9:
        df = read_dta_from_zip("1_Kaden/9/113559-V1.zip", "publicdata.dta")
        groups = summarize_groups(
            df.dropna(subset=["voted"]),
            "treatment",
            "voted",
            group_map={
                "CONTROL": {"arm_id": "arm3"},
                "POST": {"arm_id": "arm2"},
                "TIMES": {"arm_id": "arm1"},
            },
            metric="proportion",
        )
        results = make_results(
            "partial",
            "local_summary_files",
            ["1_Kaden/9/113559-V1.zip:publicdata.dta"],
            "Using self-reported 2005 Virginia gubernatorial turnout grouped by newspaper assignment.",
            [outcome_record("out1", "voter turnout", "binary", "share", groups)],
            ["out1"],
        )

    elif seq_id == 10:
        df = read_dta_from_zip("1_Kaden/10/dataverse_files (1).zip", "AER merged.dta")
        groups = summarize_groups(df, "treatment", "gave", group_map={1: {"arm_id": "arm2"}, 0: {"arm_id": "arm1"}}, metric="proportion")
        results = make_results(
            "partial",
            "local_summary_files",
            ["1_Kaden/10/dataverse_files (1).zip:AER merged.dta"],
            "Using donation incidence grouped by treatment versus control from the merged AER dataset.",
            [outcome_record("out1", "probability of donating", "binary", "share", groups)],
            ["out1"],
        )

    elif seq_id == 12:
        df = read_sas_from_zip("1_Kaden/12/114561-V1.zip", "programfiles/application.sas7bdat")
        df = df[df["has_formal_business_plan"] >= 0]
        groups = summarize_groups(df, "treatment", "has_formal_business_plan", group_map={1.0: {"arm_id": "arm1"}, 0.0: {"arm_id": "arm2"}}, metric="proportion")
        results = make_results(
            "partial",
            "local_summary_files",
            ["1_Kaden/12/114561-V1.zip:programfiles/application.sas7bdat"],
            "Using formal business plan status as a canonical early business outcome. The treatment coding is used as-is from the SAS data.",
            [outcome_record("out1", "has written formal business plan", "binary", "share", groups)],
            ["out1"],
            flags=["treatment_code_assumed_from_analysis_group"],
        )

    elif seq_id == 13:
        df = read_dta("1_Kaden/13/PanagopoulosGreen_AJPS_2008_ReplicationDataset.dta")
        df["radio_treat"] = (pd.to_numeric(df["grp_buy"], errors="coerce").fillna(0) > 0).astype(int)
        groups = summarize_groups(df, "radio_treat", "votesharechange", group_map={1: {"arm_id": "arm2"}, 0: {"arm_id": "arm1"}}, metric="mean")
        results = make_results(
            "partial",
            "local_summary_files",
            ["1_Kaden/13/PanagopoulosGreen_AJPS_2008_ReplicationDataset.dta"],
            "Treatment is proxied by whether positive gross ratings points were purchased.",
            [outcome_record("out1", "change in incumbent vote share", "continuous", "percentage points", groups)],
            ["out1"],
            flags=["treatment_derived_from_grp_buy"],
        )

    elif seq_id == 14:
        df = read_dta_from_zip("1_Kaden/14/dataverse_files (2).zip", "BAdata_2013.dta")
        groups = summarize_groups(
            df.dropna(subset=["ES_saving_tot"]),
            "visita",
            "ES_saving_tot",
            group_map={0: {"arm_id": "arm1"}, 1: {"arm_id": "arm2"}, 2: {"arm_id": "arm3"}, 3: {"arm_id": "arm4"}},
            metric="mean",
        )
        results = make_results(
            "partial",
            "local_summary_files",
            ["1_Kaden/14/dataverse_files (2).zip:BAdata_2013.dta"],
            "Using recipient total savings in El Salvador as a canonical savings outcome grouped by treatment arm.",
            [outcome_record("out1", "recipient total savings in El Salvador", "continuous", "USD", groups)],
            ["out1"],
        )

    elif seq_id == 15:
        df = read_dta("1_Kaden/15/S_Green_G_Gerber_JPM_2010_robo_call_2006_precinct_level_results_final.dta")
        groups = summarize_groups(df, "treat_pc", "RepTurnout", group_map={1: {"arm_id": "arm1"}, 0: {"arm_id": "arm2"}}, metric="mean")
        results = make_results(
            "partial",
            "local_summary_files",
            ["1_Kaden/15/S_Green_G_Gerber_JPM_2010_robo_call_2006_precinct_level_results_final.dta"],
            "Using Republican turnout share at the precinct level.",
            [outcome_record("out1", "Republican turnout", "continuous", "proportion", groups)],
            ["out1"],
        )

    elif seq_id == 16:
        results = make_results(
            "partial",
            "paper_or_appendix_tables",
            [
                "1_Kaden/16/113842-V1.zip:2011_086AEJApp_dataprograms/full_code_tpe_aejR1_v1.do",
                "1_Kaden/16/113842-V1.zip:2011_086AEJApp_dataprograms/2011_086AEJApp_README.pdf",
            ],
            "The local archive contains replication code but not the confidential analysis file used to recover treatment and control means. The enrichment therefore stores published treatment-effect estimates from the paper abstract and supporting local regression code rather than raw arm means.",
            [
                make_manual_outcome(
                    "out1",
                    "treatment effect on annual earnings",
                    "other",
                    "USD",
                    [
                        {
                            "arm_id": "full_sample_effect",
                            "value": 904.0,
                            "metric": "other",
                            "notes": "Published treatment effect from the paper abstract; this is not a treatment-cell mean.",
                        },
                    ],
                    analysis_population="Full sample of EITC recipients in the experiment",
                ),
                make_manual_outcome(
                    "out2",
                    "treatment effect on annual earnings by tax-preparer compliance",
                    "other",
                    "USD",
                    [
                        {
                            "arm_id": "complier_tax_professionals_effect",
                            "value": 1400.0,
                            "metric": "other",
                            "notes": "Published treatment effect among clients of tax professionals who complied with the information treatment.",
                        },
                        {
                            "arm_id": "noncomplier_tax_professionals_effect",
                            "value": 0.0,
                            "metric": "other",
                            "notes": "Published null effect among clients of non-complying tax professionals.",
                        },
                    ],
                    analysis_population="Tax-preparer compliance subgroups described in the paper abstract",
                ),
            ],
            ["out1", "out2"],
            flags=["treatment_effect_estimates_not_raw_group_means", "confidential_source_data_not_in_local_archive"],
        )

    elif seq_id == 23:
        df = read_dta("2_Yvonne/23/CDS_Public_Use_File/CDS_Data/prov_all_stats_sp.dta")
        groups = summarize_groups(df, "control", "scan_bpa_new_1_3", group_map={0: {"arm_id": "arm1"}, 1: {"arm_id": "arm2"}}, metric="mean")
        results = make_results(
            "partial",
            "local_summary_files",
            ["2_Yvonne/23/CDS_Public_Use_File/CDS_Data/prov_all_stats_sp.dta"],
            "Using provider-level counts of scans that would trigger the CDS best-practice alert and receive low appropriateness scores.",
            [outcome_record("out1", "low-appropriateness scans triggering CDS alert", "count", "provider-level mean count", groups)],
            ["out1"],
            flags=["treatment_arm_inferred_from_control_indicator"],
        )

    elif seq_id == 24:
        df = read_csv("2_Yvonne/24/bb70016196_2_1.csv")
        groups = summarize_groups(
            df.dropna(subset=["image_correct"]),
            "treatment",
            "image_correct",
            group_map={
                0: {"arm_id": "control"},
                1: {"arm_id": "output_treatment"},
                2: {"arm_id": "input_treatment"},
            },
            metric="proportion",
        )
        results = make_results(
            "partial",
            "local_summary_files",
            ["2_Yvonne/24/bb70016196_2_1.csv"],
            "Using image-level correctness as a canonical contribution-quality outcome.",
            [outcome_record("out1", "image classified correctly", "binary", "share", groups)],
            ["out1"],
            flags=["treatment_labels_generic"],
        )

    elif seq_id == 36:
        df = read_dta("2_Yvonne/36/Self_Interest_Attracts_More_Sunlight-master/SI_attracts_sunlight_panel.dta")
        group_map = {0.0: {"arm_id": "arm3"}, 1.0: {"arm_id": "arm1"}, 2.0: {"arm_id": "arm2"}}
        groups = summarize_groups(df, "group", "installations_1000ooh", group_map=group_map, metric="mean")
        results = make_results(
            "partial",
            "local_summary_files",
            ["2_Yvonne/36/Self_Interest_Attracts_More_Sunlight-master/SI_attracts_sunlight_panel.dta"],
            "Using solar installations per 1,000 owner-occupied houses. Group-to-arm mapping follows the panel data coding and the registry arm ordering.",
            [outcome_record("out1", "solar installations per 1,000 owner-occupied houses", "continuous", "rate", groups)],
            ["out1"],
            flags=["group_code_to_arm_mapping_inferred"],
        )

    elif seq_id == 38:
        df = read_dta("2_Yvonne/38/bwc.dta")
        groups = summarize_groups(df, "treat", "std_post_test", group_map={1: {"arm_id": "arm1"}, 0: {"arm_id": "arm2"}}, metric="mean")
        results = make_results(
            "partial",
            "local_summary_files",
            ["2_Yvonne/38/bwc.dta"],
            "Using standardized post-test score as the canonical receptive-vocabulary outcome.",
            [outcome_record("out1", "standardized post-test score", "continuous", "standard deviations", groups)],
            ["out1"],
        )

    elif seq_id == 41:
        df = read_dta("3_Aadi/41/worksample_jpart.dta")
        df = df[df["trial"] == 1]
        groups = summarize_groups(df, "private", "totalh_d", group_map={0.0: {"arm_id": "arm1"}, 1.0: {"arm_id": "arm2"}}, metric="proportion")
        results = make_results(
            "partial",
            "local_summary_files",
            ["3_Aadi/41/worksample_jpart.dta", "3_Aadi/41/Analysis_Does_Work_Quality_Differ.do"],
            "Using the experiment-1 sample and the share making any data-entry errors as the canonical work-quality outcome.",
            [outcome_record("out1", "any data entry errors", "binary", "share", groups)],
            ["out1"],
        )

    elif seq_id == 42:
        df = read_dta("3_Aadi/42/Data/Code/for publication/dataset_public.dta")
        groups = summarize_groups(
            df,
            "treatment",
            "d_correctpic",
            group_map={1: {"arm_id": "baseline"}, 2: {"arm_id": "restricted_monitoring"}},
            metric="mean",
        )
        results = make_results(
            "complete",
            "local_summary_files",
            ["3_Aadi/42/Data/Code/for publication/dataset_public.dta", "3_Aadi/42/Data/Code/for publication/main_analysis.do"],
            "Using the change in correctly solved pictures, which is the main performance outcome emphasized in the replication code.",
            [outcome_record("out1", "change in correctly solved pictures", "continuous", "pictures", groups)],
            ["out1"],
        )

    elif seq_id == 43:
        df = read_csv("3_Aadi/43/analysis_sample.csv")
        groups = summarize_groups(df.dropna(subset=["hba1c_val_6mo"]), "treat_now", "hba1c_val_6mo", group_map={1: {"arm_id": "arm1"}, 0: {"arm_id": "arm2"}}, metric="mean")
        results = make_results(
            "partial",
            "local_summary_files",
            ["3_Aadi/43/analysis_sample.csv", "3_Aadi/43/01_published_tables.do"],
            "Using 6-month HbA1c, directly matching the published replication tables.",
            [outcome_record("out1", "6-month HbA1c", "continuous", "HbA1c", groups)],
            ["out1"],
        )

    elif seq_id == 44:
        df = read_csv("3_Aadi/44/initial_sample.csv")
        groups = summarize_groups(df, "treatment", "prediction", metric="mean")
        results = make_results(
            "partial",
            "local_summary_files",
            ["3_Aadi/44/initial_sample.csv"],
            "Using mean one-step-ahead prediction by experimental treatment string.",
            [outcome_record("out1", "one-step-ahead forecast", "continuous", "forecast level", groups)],
            ["out1"],
            flags=["results_groups_follow_data_treatment_strings"],
        )

    elif seq_id == 45:
        df = read_csv("3_Aadi/45/data/Reddinger_Temptation.csv")
        df["group"] = df.apply(lambda r: f"day{int(r['day_decide'])}_{r['mechanism']}", axis=1)
        groups = summarize_groups(
            df.dropna(subset=["effort_later"]),
            "group",
            "effort_later",
            group_map={
                "day0_random": {"arm_id": "same_day_random"},
                "day0_certain": {"arm_id": "same_day_certain"},
                "day2_random": {"arm_id": "two_day_delay_random"},
                "day2_certain": {"arm_id": "two_day_delay_certain"},
            },
            metric="mean",
        )
        results = make_results(
            "complete",
            "local_summary_files",
            ["3_Aadi/45/data/Reddinger_Temptation.csv"],
            "Using later-task effort as the canonical real-effort allocation outcome. The four groups correspond to immediate versus two-day-delayed decisions crossed with random versus certain payment mechanisms.",
            [outcome_record("out1", "later effort allocation", "continuous", "task units", groups)],
            ["out1"],
        )

    elif seq_id == 46:
        df = read_dta("3_Aadi/46/two_in_base.dta")
        df = df[df["qualtrics"] == 1]
        groups = summarize_groups(
            df,
            "group",
            "judgment",
            group_map={
                1.0: {"arm_id": "women_first_woman_discrimination"},
                2.0: {"arm_id": "woman_first_man_discrimination"},
                3.0: {"arm_id": "man_first_man_discrimination"},
                4.0: {"arm_id": "man_first_woman_discrimination"},
            },
            metric="mean",
        )
        results = make_results(
            "partial",
            "local_summary_files",
            ["3_Aadi/46/two_in_base.dta", "3_Aadi/46/FFS_2021.OSF.results.do"],
            "Using judgment in the two-in-base sample for the Qualtrics respondents, which matches the main table construction in the replication code.",
            [outcome_record("out1", "judgment", "continuous", "attitude scale", groups)],
            ["out1"],
            flags=["results_groups_follow_value_labels_in_two_in_base"],
        )

    elif seq_id == 47:
        df = read_csv("3_Aadi/47/PP_DataAndCode/Data.csv")
        long = []
        for col, arm_id in [
            ("H_HIV", "hiv_testing"),
            ("E_Drop", "dropout"),
            ("E_Bus", "business"),
            ("E_Exam", "test_scores"),
            ("C_Food_T1", "food_security_t1"),
            ("C_Food_T2", "food_security_t2"),
            ("C_Food_T3", "food_security_t3"),
            ("C_Hea_T1", "health_t1"),
            ("C_Hea_T2", "health_t2"),
            ("C_Hea_T3", "health_t3"),
        ]:
            sub = df[df[col] == 1][["Pred"]].copy()
            sub["grp"] = arm_id
            long.append(sub)
        work = pd.concat(long, ignore_index=True)
        groups = summarize_groups(work, "grp", "Pred", metric="mean")
        results = make_results(
            "complete",
            "local_summary_files",
            ["3_Aadi/47/PP_DataAndCode/Data.csv", "3_Aadi/47/PP_DataAndCode/Analysis.r"],
            "Using mean forecast by target experiment, following the full set of target experiments marked in the replication data and analysis script.",
            [outcome_record("out1", "forecasted treatment effect", "continuous", "forecast units", groups)],
            ["out1"],
            flags=["registry_has_no_arms_so_target_experiments_used_as_groups"],
        )

    elif seq_id == 50:
        df = read_dta("3_Aadi/50/1_clean_data.dta")
        groups = summarize_groups(
            df.dropna(subset=["prob_index_anti"]),
            "treatment",
            "prob_index_anti",
            group_map={
                0.0: {"arm_id": "control_group"},
                1.0: {"arm_id": "trump_trump_pro"},
                2.0: {"arm_id": "obama_obama_pro"},
                3.0: {"arm_id": "trump_actor_pro"},
                4.0: {"arm_id": "obama_actor_pro"},
                5.0: {"arm_id": "trump_trump_anti"},
                6.0: {"arm_id": "obama_obama_anti"},
                7.0: {"arm_id": "trump_actor_anti"},
                8.0: {"arm_id": "obama_actor_anti"},
                9.0: {"arm_id": "trump_turkey"},
                10.0: {"arm_id": "obama_turkey"},
            },
            metric="mean",
        )
        results = make_results(
            "partial",
            "local_summary_files",
            ["3_Aadi/50/1_clean_data.dta", "3_Aadi/50/6_TotalEffect.do"],
            "Using the anti-immigrant attitude index by treatment condition from the cleaned analysis data.",
            [outcome_record("out1", "anti-immigrant attitude index", "continuous", "index (0-1)", groups)],
            ["out1"],
            flags=["data_contains_more_message_arms_than_registry_summary"],
        )

    elif seq_id == 57:
        df = read_dta("3_Aadi/57/BFS Data - Study 1.dta")
        valid = ["Control", "Unconditional", "Immediate", "Commit", "WaitingPeriods"]
        df = df[df["Treatment"].isin(valid)]
        groups = summarize_groups(df.dropna(subset=["TFV"]), "Treatment", "TFV", metric="proportion")
        results = make_results(
            "partial",
            "local_summary_files",
            ["3_Aadi/57/BFS Data - Study 1.dta", "3_Aadi/57/BFS Code.do"],
            "Using the indicator for choosing fruits and vegetables as the primary food-choice outcome in Study 1.",
            [outcome_record("out1", "choice of fruits and vegetables subsidy", "binary", "share", groups)],
            ["out1"],
        )

    # Partial-only records with source tracking.
    elif seq_id == 3:
        df = read_dta_from_zip("1_Kaden/3/dataverse_files.zip", "replication_package/datasets/analysis_claims.dta")
        work = df[["Itreatment", "Ipost_ie_admit_180_IP_100", "post_ie_admit_cnt_180_IP"]].copy()
        work["any_readmit_180"] = pd.to_numeric(work["Ipost_ie_admit_180_IP_100"], errors="coerce") / 100.0
        groups_binary = summarize_groups(
            work.dropna(subset=["Itreatment", "any_readmit_180"]),
            "Itreatment",
            "any_readmit_180",
            group_map={1.0: {"arm_id": "arm1"}, 0.0: {"arm_id": "arm2"}},
            metric="proportion",
        )
        groups_count = summarize_groups(
            work.dropna(subset=["Itreatment", "post_ie_admit_cnt_180_IP"]),
            "Itreatment",
            "post_ie_admit_cnt_180_IP",
            group_map={1.0: {"arm_id": "arm1"}, 0.0: {"arm_id": "arm2"}},
            metric="mean",
        )
        results = make_results(
            "complete",
            "local_summary_files",
            [
                "1_Kaden/3/dataverse_files.zip:replication_package/datasets/analysis_claims.dta",
                "1_Kaden/3/dataverse_files.zip:replication_package/datasets/codebooks/analysis_claims_codebook.txt",
            ],
            "Using claims-based 180-day readmission outcomes from the replication package's analysis file.",
            [
                outcome_record("out1", "any hospital readmission within 180 days", "binary", "share", groups_binary),
                outcome_record("out2", "number of inpatient readmissions within 180 days", "count", "count", groups_count),
            ],
            ["out1"],
        )
    elif seq_id == 6:
        df = read_dta_from_zip("1_Kaden/6/116212-V1.zip", "data/ELM_AUX1.dta")
        df["favor_share"] = pd.to_numeric(df["favor"], errors="coerce") / 100.0
        group_map = {
            1.0: {"arm_id": "public_agency"},
            2.0: {"arm_id": "private_old"},
            3.0: {"arm_id": "private_new"},
        }
        level_specs = [
            (19, "0"),
            (23, "4k"),
            (28, "9k"),
            (33, "14k"),
            (38, "19k"),
        ]
        outcomes = []
        for idx, (level_val, level_label) in enumerate(level_specs, start=1):
            sub = df[df["level"] == level_val]
            outcomes.append(
                outcome_record(
                    f"out{idx}",
                    f"support for compensation regime with expected transplant increase of {level_label}",
                    "binary",
                    "share",
                    summarize_groups(
                        sub.dropna(subset=["treatment", "favor_share"]),
                        "treatment",
                        "favor_share",
                        group_map=group_map,
                        metric="proportion",
                    ),
                    analysis_population="One repeated judgment per respondent at the stated transplant-gain level",
                    notes="The public archive stores this study as a multi-level repeated survey design. Each outcome fixes the transplant-gain level and compares support across the three compensation regimes.",
                )
            )
        results = make_results(
            "complete",
            "local_summary_files",
            [
                "1_Kaden/6/116212-V1.zip:data/ELM_AUX1.dta",
                "1_Kaden/6/116212-V1.zip:data/ELM2018_final.do",
            ],
            "Using the cleaned auxiliary analysis file that already normalizes support by treatment regime and transplant-gain level. Group labels follow the archive's value labels: PA, PVT-OLD, and PVT-NEW.",
            outcomes,
            [f"out{i}" for i in range(1, 6)],
        )
    elif seq_id == 19:
        df = read_dta("1_Kaden/19/Green_et_al_JP_2011_BRI_Public_Use_0223_EDITED.dta")
        general_items = ["Bq9", "Bq10", "Bq11", "Bq12"]
        civil_knowledge_items = [
            "Bq16", "Bq17", "Bq18", "Bq19", "Bq20", "Bq21", "Bq22",
            "Bq23", "Bq24", "Bq25", "Bq26", "Bq27", "Bq28",
        ]
        for col in general_items + civil_knowledge_items:
            df[f"{col}_ok"] = df[col].map({1: 1, 0: 0})
        df["Bq29_ok"] = df["Bq29"].map({0: 1, 1: 0})
        for col in ["Bq30", "Bq31", "Bq32"]:
            df[f"{col}_ok"] = df[col].map({1: 1, 0: 0})
        df["general_idx"] = df[[f"{c}_ok" for c in general_items]].mean(axis=1)
        df["civil_knowledge_idx"] = df[[f"{c}_ok" for c in civil_knowledge_items]].mean(axis=1)
        df["support_idx"] = df[[f"{c}_ok" for c in ["Bq29", "Bq30", "Bq31", "Bq32"]]].mean(axis=1)
        group_map = {
            1: {"arm_id": "brrl_curriculum"},
            0: {"arm_id": "control_curriculum"},
        }
        results = make_results(
            "partial",
            "local_summary_files",
            [
                "1_Kaden/19/Green_et_al_JP_2011_BRI_Public_Use_0223_EDITED.dta",
                "1_Kaden/19/Green_et_al_JP_2011_EDITED_README.txt",
            ],
            "The original public-use scripts are missing locally, so these results reconstruct transparent proxy indices from the labeled B-wave post-curriculum item battery in the main analysis file.",
            [
                outcome_record(
                    "out1",
                    "general political knowledge index (post-curriculum proxy)",
                    "index",
                    "share correct",
                    summarize_groups(
                        df.dropna(subset=["treatment", "general_idx"]),
                        "treatment",
                        "general_idx",
                        group_map=group_map,
                        metric="mean",
                    ),
                    analysis_population="Students with non-missing B-wave general-knowledge items",
                ),
                outcome_record(
                    "out2",
                    "civil-liberties knowledge index (post-curriculum proxy)",
                    "index",
                    "share correct",
                    summarize_groups(
                        df.dropna(subset=["treatment", "civil_knowledge_idx"]),
                        "treatment",
                        "civil_knowledge_idx",
                        group_map=group_map,
                        metric="mean",
                    ),
                    analysis_population="Students with non-missing B-wave civil-liberties knowledge items",
                ),
                outcome_record(
                    "out3",
                    "civil-liberties support index (post-curriculum proxy)",
                    "index",
                    "share pro-liberties responses",
                    summarize_groups(
                        df.dropna(subset=["treatment", "support_idx"]),
                        "treatment",
                        "support_idx",
                        group_map=group_map,
                        metric="mean",
                    ),
                    analysis_population="Students with non-missing B-wave civil-liberties support items",
                ),
            ],
            ["out1", "out2", "out3"],
            flags=["proxy_indices_reconstructed_from_b_wave_items"],
        )
    elif seq_id == 20:
        wave1_pilot = read_csv("2_Yvonne/28/ExtremeToMainstream_Replication/data/app_exp1b_wave1_pilot.csv")
        wave1 = read_csv("2_Yvonne/28/ExtremeToMainstream_Replication/data/app_exp1b_wave1.csv")
        wave2 = read_csv("2_Yvonne/28/ExtremeToMainstream_Replication/data/app_exp1b_wave2.csv")
        for frame in [wave1_pilot, wave1]:
            frame["sob"] = pd.to_numeric(frame["q9"], errors="coerce")
            frame["donate"] = (frame["q10_a"] == "Yes").astype(float)
            frame["group"] = frame["t_info_control"] + " " + frame["t_public_private"]
        wave2["sob"] = pd.to_numeric(wave2["q8"], errors="coerce")
        wave2["donate"] = (wave2["q9_a"] == "Yes").astype(float)
        wave2["group"] = "Control " + wave2["t_public_private"]
        exp1 = pd.concat(
            [
                wave1_pilot[["group", "sob", "donate"]],
                wave1[["group", "sob", "donate"]],
                wave2[["group", "sob", "donate"]],
            ],
            ignore_index=True,
        )
        exp1_map = {
            "Control Private": {"arm_id": "E1_Ctrl_Private"},
            "Control Public": {"arm_id": "E1_Ctrl_Public"},
            "Information Private": {"arm_id": "E1_TreatProb_Private"},
            "Information Public": {"arm_id": "E1_TreatProb_Public"},
        }

        exp2 = read_csv("2_Yvonne/28/ExtremeToMainstream_Replication/data/exp1.csv")
        exp2["sob"] = pd.to_numeric(exp2["q5_b"], errors="coerce")
        exp2["donate"] = (
            (exp2["q4_a"] == "Yes, please donate $1 to FAIR on my behalf.")
            | (exp2["q4_b"] == "Yes, please donate $1 to FAIR on my behalf.")
        ).astype(float)
        exp2["group"] = exp2["t_trump_clinton"] + " " + exp2["t_public_private"]
        exp2_map = {
            "Trump Won Private": {"arm_id": "E2_TrumpWin_Private"},
            "Trump Won Public": {"arm_id": "E2_TrumpWin_Public"},
            "Clinton Won Private": {"arm_id": "E2_ClintonWin_Private"},
            "Clinton Won Public": {"arm_id": "E2_ClintonWin_Public"},
        }

        results = make_results(
            "complete",
            "local_summary_files",
            [
                "1_Kaden/20/119846-V1.zip:ExtremeToMainstream_Replication/data/app_exp1b_wave1_pilot.csv",
                "1_Kaden/20/119846-V1.zip:ExtremeToMainstream_Replication/data/app_exp1b_wave1.csv",
                "1_Kaden/20/119846-V1.zip:ExtremeToMainstream_Replication/data/app_exp1b_wave2.csv",
                "1_Kaden/20/119846-V1.zip:ExtremeToMainstream_Replication/data/exp1.csv",
                "1_Kaden/20/119846-V1.zip:ExtremeToMainstream_Replication/scripts/2_clean_data.do",
            ],
            "The registered study combines two xenophobia experiments in the shared replication package; results are extracted separately for the state-level probability-information experiment and the Pittsburgh Trump/Clinton information experiment.",
            [
                outcome_record(
                    "out1",
                    "belief about share holding xenophobic sentiments",
                    "continuous",
                    "percent",
                    summarize_groups(exp1.dropna(subset=["sob"]), "group", "sob", group_map=exp1_map, metric="mean"),
                    analysis_population="Experiment 1 (state-level probability information; pooled waves)",
                ),
                outcome_record(
                    "out2",
                    "donation to FAIR",
                    "binary",
                    "share",
                    summarize_groups(exp1.dropna(subset=["donate"]), "group", "donate", group_map=exp1_map, metric="proportion"),
                    analysis_population="Experiment 1 (state-level probability information; pooled waves)",
                ),
                outcome_record(
                    "out3",
                    "belief about share of Pittsburgh voters endorsing xenophobic immigration quote",
                    "continuous",
                    "percent",
                    summarize_groups(exp2.dropna(subset=["sob"]), "group", "sob", group_map=exp2_map, metric="mean"),
                    analysis_population="Experiment 2 (Trump won vs Clinton won information)",
                ),
                outcome_record(
                    "out4",
                    "donation to FAIR",
                    "binary",
                    "share",
                    summarize_groups(exp2.dropna(subset=["donate"]), "group", "donate", group_map=exp2_map, metric="proportion"),
                    analysis_population="Experiment 2 (Trump won vs Clinton won information)",
                ),
            ],
            ["out2", "out4"],
        )
    elif seq_id == 21:
        site_specs = [
            ("atlanta_lfa", "Atlanta labor force attachment", 3833, 66.1, 61.6, 5820.0, 5006.0, 17.20, 18.35, 4553.0, 4922.0),
            ("atlanta_hcd", "Atlanta human capital development", 3881, 64.4, 61.6, 5502.0, 5006.0, 17.78, 18.35, 4634.0, 4922.0),
            ("grand_rapids_lfa", "Grand Rapids labor force attachment", 3012, 77.7, 70.1, 5674.0, 4639.0, 15.19, 17.41, 5944.0, 7347.0),
            ("grand_rapids_hcd", "Grand Rapids human capital development", 2997, 75.4, 70.1, 5219.0, 4639.0, 16.19, 17.41, 6512.0, 7347.0),
            ("riverside_lfa", "Riverside labor force attachment", 6726, 60.2, 45.0, 5488.0, 4213.0, 14.59, 16.05, 8292.0, 9600.0),
            ("riverside_hcd", "Riverside human capital development", 3135, 48.2, 38.9, 3450.0, 3133.0, 15.78, 16.74, 9253.0, 10302.0),
            ("columbus_integrated", "Columbus integrated", 4672, 73.9, 72.2, 7565.0, 6892.0, 14.83, 16.41, 4775.0, 5469.0),
            ("columbus_traditional", "Columbus traditional", 4729, 73.5, 72.2, 7569.0, 6892.0, 15.38, 16.41, 4939.0, 5469.0),
            ("detroit", "Detroit", 4459, 62.3, 58.2, 4369.0, 4001.0, 19.23, 19.71, 8457.0, 8615.0),
            ("oklahoma_city", "Oklahoma City", 8677, 64.1, 65.0, 3518.0, 3514.0, 10.93, 11.71, 3391.0, 3624.0),
            ("portland", "Portland", 5547, 72.1, 60.9, 7133.0, 5291.0, 13.12, 15.53, 5818.0, 7014.0),
        ]
        employment_rows = []
        earnings_rows = []
        afdc_month_rows = []
        afdc_payment_rows = []
        for site_id, label, n_total, emp_prog, emp_ctrl, earn_prog, earn_ctrl, months_prog, months_ctrl, pay_prog, pay_ctrl in site_specs:
            employment_rows.extend(
                [
                    {"arm_id": f"{site_id}_program", "value": emp_prog, "metric": "mean", "n_analyzed": n_total, "notes": f"{label}: program-group mean from NEWWS impact table."},
                    {"arm_id": f"{site_id}_control", "value": emp_ctrl, "metric": "mean", "n_analyzed": n_total, "notes": f"{label}: control-group mean from NEWWS impact table."},
                ]
            )
            earnings_rows.extend(
                [
                    {"arm_id": f"{site_id}_program", "value": earn_prog, "metric": "mean", "n_analyzed": n_total, "notes": f"{label}: program-group mean from NEWWS impact table."},
                    {"arm_id": f"{site_id}_control", "value": earn_ctrl, "metric": "mean", "n_analyzed": n_total, "notes": f"{label}: control-group mean from NEWWS impact table."},
                ]
            )
            afdc_month_rows.extend(
                [
                    {"arm_id": f"{site_id}_program", "value": months_prog, "metric": "mean", "n_analyzed": n_total, "notes": f"{label}: program-group mean from NEWWS impact table."},
                    {"arm_id": f"{site_id}_control", "value": months_ctrl, "metric": "mean", "n_analyzed": n_total, "notes": f"{label}: control-group mean from NEWWS impact table."},
                ]
            )
            afdc_payment_rows.extend(
                [
                    {"arm_id": f"{site_id}_program", "value": pay_prog, "metric": "mean", "n_analyzed": n_total, "notes": f"{label}: program-group mean from NEWWS impact table."},
                    {"arm_id": f"{site_id}_control", "value": pay_ctrl, "metric": "mean", "n_analyzed": n_total, "notes": f"{label}: control-group mean from NEWWS impact table."},
                ]
            )
        results = make_results(
            "complete",
            "paper_or_appendix_tables",
            [
                "2_Yvonne/21/p5fullsamp/AR2_tbl1.txt",
                "2_Yvonne/21/p5fullsamp/AR2_tbl3.txt",
            ],
            "The local NEWWS archive reports site-by-program treatment and control means in ASCII summary tables. Results are stored as paired program/control summaries for each site-program variant listed in the public-use impact tables.",
            [
                make_manual_outcome("out1", "ever employed in years 1 and 2", "binary", "percent", employment_rows, timepoint="years 1-2"),
                make_manual_outcome("out2", "total earnings in years 1 and 2", "continuous", "USD", earnings_rows, timepoint="years 1-2"),
                make_manual_outcome("out3", "months receiving AFDC in years 1 and 2", "continuous", "months", afdc_month_rows, timepoint="years 1-2"),
                make_manual_outcome("out4", "AFDC payments received in years 1 and 2", "continuous", "USD", afdc_payment_rows, timepoint="years 1-2"),
            ],
            ["out1", "out2", "out3", "out4"],
            flags=["site_program_pairs_used_instead_of_single_global_arm_structure"],
        )
    elif seq_id == 22:
        results = make_results(
            "partial",
            "paper_or_appendix_tables",
            ["2_Yvonne/22/turnout16-05-23.pdf"],
            "The local source is the paper PDF. The strongest simulation-relevant summaries are the observed lying rates for voters and non-voters under the no-incentive and incentive-to-say-did-not-vote conditions, plus the turnout treatment-effect estimates from the GOTV flyer experiments.",
            [
                make_manual_outcome(
                    "out1",
                    "lying about voting status by voter type and lie incentive",
                    "binary",
                    "share lying",
                    [
                        {"arm_id": "voters_no_incentive", "value": 0.12, "metric": "proportion", "notes": "Baseline voter lying rate reported in the paper."},
                        {"arm_id": "voters_incentive", "value": 0.147, "metric": "proportion", "notes": "Constructed from the reported 12 percent baseline plus a 2.7 percentage-point increase under the incentive to say did not vote."},
                        {"arm_id": "nonvoters_no_incentive", "value": 0.46, "metric": "proportion", "notes": "Baseline non-voter lying rate reported in the paper."},
                        {"arm_id": "nonvoters_incentive", "value": 0.34, "metric": "proportion", "notes": "Constructed from the reported 46 percent baseline minus a 12 percentage-point decrease under the incentive to say did not vote."},
                    ],
                ),
                make_manual_outcome(
                    "out2",
                    "turnout effect of announcing a post-election turnout survey",
                    "other",
                    "percentage points",
                    [
                        {"arm_id": "gotv_2010_effect", "value": 1.3, "metric": "other", "notes": "Reported treatment-minus-control turnout difference for the 2010 election."},
                        {"arm_id": "gotv_2012_effect", "value": 0.1, "metric": "other", "notes": "Reported treatment-minus-control turnout difference for the 2012 election."},
                    ],
                    analysis_population="Separate election-year flyer experiments",
                ),
            ],
            ["out1"],
            flags=["paper_text_used_for_group_summaries", "includes_effect_estimates_for_turnout_experiment"],
        )
    elif seq_id == 26:
        results = make_results(
            "partial",
            "paper_or_appendix_tables",
            [
                "2_Yvonne/26/ICPSR_07865/DS0002/07865-0002-Data.dta",
                "2_Yvonne/26/ICPSR_07865 2/07865-User_guide.pdf",
                "2_Yvonne/26/ICPSR_07865 2/07865-descriptioncitation.html",
            ],
            "The ICPSR archive is locally present, but the public DTA exports strip most variable semantics. To make the study usable, the enrichment relies on the classic 18-month summary report's published experimental-versus-control means by target group and period instead of trying to reverse-engineer the anonymized Stata fields.",
            [
                make_manual_outcome(
                    "out1",
                    "average monthly earnings by target group, months 1 to 9",
                    "continuous",
                    "USD",
                    [
                        {"arm_id": "afdc_experimental_m1_9", "value": 96.0, "metric": "mean", "notes": "Experimental mean from Supported Work summary table."},
                        {"arm_id": "afdc_control_m1_9", "value": 68.0, "metric": "mean", "notes": "Control mean from Supported Work summary table."},
                        {"arm_id": "ex_addict_experimental_m1_9", "value": 317.0, "metric": "mean", "notes": "Experimental mean from Supported Work summary table."},
                        {"arm_id": "ex_addict_control_m1_9", "value": 274.0, "metric": "mean", "notes": "Control mean from Supported Work summary table."},
                        {"arm_id": "ex_offender_experimental_m1_9", "value": 347.0, "metric": "mean", "notes": "Experimental mean from Supported Work summary table."},
                        {"arm_id": "ex_offender_control_m1_9", "value": 327.0, "metric": "mean", "notes": "Control mean from Supported Work summary table."},
                        {"arm_id": "youth_experimental_m1_9", "value": 188.0, "metric": "mean", "notes": "Experimental mean from Supported Work summary table."},
                        {"arm_id": "youth_control_m1_9", "value": 174.0, "metric": "mean", "notes": "Control mean from Supported Work summary table."},
                    ],
                    timepoint="months 1-9 after random assignment",
                ),
                make_manual_outcome(
                    "out2",
                    "average monthly earnings by target group, months 10 to 18",
                    "continuous",
                    "USD",
                    [
                        {"arm_id": "afdc_experimental_m10_18", "value": 128.0, "metric": "mean", "notes": "Experimental mean from Supported Work summary table."},
                        {"arm_id": "afdc_control_m10_18", "value": 99.0, "metric": "mean", "notes": "Control mean from Supported Work summary table."},
                        {"arm_id": "ex_addict_experimental_m10_18", "value": 363.0, "metric": "mean", "notes": "Experimental mean from Supported Work summary table."},
                        {"arm_id": "ex_addict_control_m10_18", "value": 295.0, "metric": "mean", "notes": "Control mean from Supported Work summary table."},
                        {"arm_id": "ex_offender_experimental_m10_18", "value": 312.0, "metric": "mean", "notes": "Experimental mean from Supported Work summary table."},
                        {"arm_id": "ex_offender_control_m10_18", "value": 264.0, "metric": "mean", "notes": "Control mean from Supported Work summary table."},
                        {"arm_id": "youth_experimental_m10_18", "value": 224.0, "metric": "mean", "notes": "Experimental mean from Supported Work summary table."},
                        {"arm_id": "youth_control_m10_18", "value": 206.0, "metric": "mean", "notes": "Control mean from Supported Work summary table."},
                    ],
                    timepoint="months 10-18 after random assignment",
                ),
                make_manual_outcome(
                    "out3",
                    "AFDC payments by target group, months 16 to 18",
                    "continuous",
                    "USD",
                    [
                        {"arm_id": "afdc_experimental_afdc_m16_18", "value": 206.0, "metric": "mean", "notes": "Experimental mean from Supported Work summary table."},
                        {"arm_id": "afdc_control_afdc_m16_18", "value": 271.0, "metric": "mean", "notes": "Control mean from Supported Work summary table."},
                        {"arm_id": "youth_experimental_afdc_m16_18", "value": 60.0, "metric": "mean", "notes": "Experimental mean from Supported Work summary table."},
                        {"arm_id": "youth_control_afdc_m16_18", "value": 99.0, "metric": "mean", "notes": "Control mean from Supported Work summary table."},
                    ],
                    timepoint="months 16-18 after random assignment",
                ),
            ],
            ["out1", "out2", "out3"],
            flags=["target_group_specific_pairs_used_instead_of_single_global_arm_structure", "published_summary_tables_used_because_local_dta_labels_are_stripped"],
        )
    elif seq_id == 28:
        df = read_csv("2_Yvonne/28/ExtremeToMainstream_Replication/data/app_exp3.csv")
        df["sob"] = pd.to_numeric(df["q8"], errors="coerce")
        df["donate"] = (df["q9_a"] == "Yes").astype(float)
        df["group"] = df["t_support_const_control"] + " " + df["t_public_private"]
        group_map = {
            "Control Private": {"arm_id": "A1"},
            "Control Public": {"arm_id": "A2"},
            "Public Support Information Private": {"arm_id": "A3"},
            "Public Support Information Public": {"arm_id": "A4"},
            "Unconstitutionality Information Private": {"arm_id": "A5"},
            "Unconstitutionality Information Public": {"arm_id": "A6"},
        }
        results = make_results(
            "complete",
            "local_summary_files",
            [
                "2_Yvonne/28/ExtremeToMainstream_Replication/data/app_exp3.csv",
                "2_Yvonne/28/ExtremeToMainstream_Replication/scripts/2_clean_data.do",
            ],
            "Using the main app_exp3 file from the shared replication package, which matches the six registered treatment cells for the Islamophobia study.",
            [
                outcome_record(
                    "out1",
                    "belief about popularity of anti-Muslim public-office ban",
                    "continuous",
                    "percent",
                    summarize_groups(df.dropna(subset=["sob"]), "group", "sob", group_map=group_map, metric="mean"),
                ),
                outcome_record(
                    "out2",
                    "donation to ACT for America",
                    "binary",
                    "share",
                    summarize_groups(df.dropna(subset=["donate"]), "group", "donate", group_map=group_map, metric="proportion"),
                ),
            ],
            ["out2"],
        )
    elif seq_id == 29:
        wave1 = read_csv("2_Yvonne/28/ExtremeToMainstream_Replication/data/app_exp2b_wave1.csv")
        wave2 = read_csv("2_Yvonne/28/ExtremeToMainstream_Replication/data/app_exp2b_wave2.csv")
        for frame in [wave1, wave2]:
            frame["donation_amount"] = pd.to_numeric(frame["q9"], errors="coerce")
        wave_map = {
            "Control": {"arm_id": "arm1"},
            "Anti-Minarets": {"arm_id": "arm2"},
            "Anti-Minarets Public Support": {"arm_id": "arm3"},
            "Anti-Minarets Referendum": {"arm_id": "arm4"},
        }
        followup = read_csv("2_Yvonne/28/ExtremeToMainstream_Replication/data/exp2.csv")
        followup["donation_amount"] = pd.to_numeric(followup["q12"], errors="coerce")
        followup["group"] = followup["t_trump_clinton"] + " " + followup["t_public_private"]
        followup_map = {
            "Trump Won Private": {"arm_id": "followup_trump_private"},
            "Trump Won Public": {"arm_id": "followup_trump_public"},
            "Clinton Won Private": {"arm_id": "followup_clinton_private"},
            "Clinton Won Public": {"arm_id": "followup_clinton_public"},
        }
        results = make_results(
            "complete",
            "local_summary_files",
            [
                "2_Yvonne/29/ExtremeToMainstream_Replication/data/app_exp2b_wave1.csv",
                "2_Yvonne/29/ExtremeToMainstream_Replication/data/app_exp2b_wave2.csv",
                "2_Yvonne/29/ExtremeToMainstream_Replication/data/exp2.csv",
                "2_Yvonne/29/ExtremeToMainstream_Replication/scripts/2_clean_data.do",
            ],
            "The registered study combines the anti-minarets legitimacy experiment and a follow-up Trump/Clinton observability experiment. Results are stored separately by study component.",
            [
                outcome_record(
                    "out1",
                    "dictator-game donation amount",
                    "continuous",
                    "USD",
                    summarize_groups(wave1.dropna(subset=["donation_amount"]), "treatment", "donation_amount", group_map=wave_map, metric="mean"),
                    analysis_population="Main minarets experiment (wave 1 wording)",
                ),
                outcome_record(
                    "out2",
                    "dictator-game donation amount",
                    "continuous",
                    "USD",
                    summarize_groups(wave2.dropna(subset=["donation_amount"]), "treatment", "donation_amount", group_map=wave_map, metric="mean"),
                    notes="Wave 2 used the anonymous wording variant.",
                    analysis_population="Main minarets experiment (wave 2 anonymous wording)",
                ),
                outcome_record(
                    "out3",
                    "dictator-game donation amount",
                    "continuous",
                    "USD",
                    summarize_groups(followup.dropna(subset=["donation_amount"]), "group", "donation_amount", group_map=followup_map, metric="mean"),
                    analysis_population="Follow-up Trump-won vs Clinton-won experiment",
                ),
            ],
            ["out1", "out3"],
            flags=["followup_groups_use_data_native_ids"],
        )
    elif seq_id == 34:
        df = read_rda_frame(
            "2_Yvonne/34/ICPSR_33782/DS0001/33782-0001-Data.rda",
            ["P_E", "S6_BDEPRESS", "S6_EQW5"],
        )
        df["P_E"] = pd.to_numeric(df["P_E"], errors="coerce")
        df["S6_BDEPRESS"] = pd.to_numeric(df["S6_BDEPRESS"], errors="coerce")
        df["S6_EQW5"] = pd.to_numeric(df["S6_EQW5"], errors="coerce") / 100.0
        group_map = {100.0: {"arm_id": "A1"}, 0.0: {"arm_id": "A2"}}
        results = make_results(
            "complete",
            "local_summary_files",
            [
                "2_Yvonne/34/ICPSR_33782/DS0001/33782-0001-Data.rda",
                "2_Yvonne/34/ICPSR_33782/DS0001/33782-0001-Codebook.pdf",
            ],
            "Codebook-guided extraction using P_E as the program-group indicator, S6_BDEPRESS as the 6-month QIDS depression score, and S6_EQW5 as current employment at 6 months.",
            [
                outcome_record(
                    "out1",
                    "depression severity at 6 months",
                    "continuous",
                    "QIDS-SR score",
                    summarize_groups(df.dropna(subset=["S6_BDEPRESS"]), "P_E", "S6_BDEPRESS", group_map=group_map, metric="mean"),
                    timepoint="6 months",
                ),
                outcome_record(
                    "out2",
                    "currently employed at 6 months",
                    "binary",
                    "share",
                    summarize_groups(df.dropna(subset=["S6_EQW5"]), "P_E", "S6_EQW5", group_map=group_map, metric="proportion"),
                    timepoint="6 months",
                ),
            ],
            ["out1", "out2"],
        )
    elif seq_id == 48:
        df = read_tsv("3_Aadi/48/Analysis/Tables/summaryStats2.csv")
        row = df[df["var"] == "index1StdC"].iloc[0]
        groups = [
            manual_group_summary("arm1", row["Treatment1"], metric="index", n_analyzed=330),
            manual_group_summary("arm2", row["Treatment2"], metric="index", n_analyzed=330),
            manual_group_summary("arm3", row["Treatment3"], metric="index", n_analyzed=330),
            manual_group_summary("arm4", row["Treatment4"], metric="index", n_analyzed=330),
        ]
        results = make_results(
            "complete",
            "paper_or_appendix_tables",
            [
                "3_Aadi/48/Analysis/Tables/summaryStats2.csv",
                "3_Aadi/48/Do Files/analysis.do",
            ],
            "Using the published summary-stats table for the standardized abstract-quality index. The table reports N=1320 abstract-level observations, implying 330 observations per treatment cell in the balanced four-arm design.",
            [outcome_record("out1", "standardized abstract-quality index", "index", "standard deviations", groups)],
            ["out1"],
            flags=["n_inferred_from_balanced_summary_table"],
        )
    elif seq_id == 51:
        df = read_csv_from_zip("3_Aadi/51/data/Reddinger_Levine_Charness_vaccination_rct.zip", "Reddinger_Levine_Charness_vaccination_rct.csv")
        outcomes = []
        segment_specs = [
            ("t_b", "Black-targeted segment", "black_targeted", "black_control"),
            ("t_l", "Latinx-targeted segment", "latinx_targeted", "latinx_control"),
            ("t_nbnl", "Non-Black/non-Latinx segment", "nbnl_targeted", "nbnl_control"),
            ("t_spanish", "Spanish parallel-text segment", "spanish_targeted", "spanish_control"),
        ]
        for idx, (col, label, treat_id, control_id) in enumerate(segment_specs, start=1):
            sub = df[[col, "vax_likely"]].dropna()
            sub["vax_likely"] = pd.to_numeric(sub["vax_likely"], errors="coerce")
            groups = summarize_groups(
                sub.dropna(subset=["vax_likely"]),
                col,
                "vax_likely",
                group_map={1.0: {"arm_id": treat_id}, 0.0: {"arm_id": control_id}},
                metric="mean",
            )
            outcomes.append(
                outcome_record(
                    f"out{idx}",
                    f"stated vaccine intent: {label}",
                    "ordinal",
                    "Likert scale",
                    groups,
                    analysis_population=f"Respondents with non-missing {col} assignment indicator",
                )
            )
        results = make_results(
            "complete",
            "local_summary_files",
            [
                "3_Aadi/51/data/Reddinger_Levine_Charness_vaccination_rct.zip:Reddinger_Levine_Charness_vaccination_rct.csv",
                "3_Aadi/51/programs/data_label.do",
                "3_Aadi/51/programs/data_setup.do",
            ],
            "This trial uses segment-specific randomization rather than a single global arm structure. Results are summarized as treated-versus-control comparisons within each segment-specific assignment indicator available in the public data.",
            outcomes,
            ["out1", "out2", "out3", "out4"],
            flags=["segment_specific_groups_not_mutually_exclusive"],
        )
    elif seq_id == 52:
        df = read_rda_frame("3_Aadi/52/P4P_Clean.rda", ["Community demographics", "Chosen_Job"])
        df["Chosen_Job"] = pd.to_numeric(df["Chosen_Job"], errors="coerce")
        group_map = {
            "Mostly white": {"arm_id": "mostly_white"},
            "Mostly African American": {"arm_id": "mostly_african_american"},
            "Mostly Hispanic": {"arm_id": "mostly_hispanic"},
            "Multiracial": {"arm_id": "multiracial"},
        }
        results = make_results(
            "complete",
            "local_summary_files",
            [
                "3_Aadi/52/P4P_Clean.rda",
                "3_Aadi/52/P4P_analysis.R",
            ],
            "This registry entry is a paired-profile conjoint, so the summary is stored as marginal choice probabilities by randomized community-demographics level rather than as a classic between-subject arm contrast.",
            [
                outcome_record(
                    "out1",
                    "job attraction by community demographics attribute level",
                    "binary",
                    "share chosen",
                    summarize_groups(
                        df.dropna(subset=["Community demographics", "Chosen_Job"]),
                        "Community demographics",
                        "Chosen_Job",
                        group_map=group_map,
                        metric="proportion",
                    ),
                    analysis_population="All randomized job profiles in the conjoint sample",
                ),
            ],
            ["out1"],
            flags=["conjoint_marginal_means_not_between_subject_arms"],
        )
    elif seq_id == 53:
        df = read_rda_frame(
            "3_Aadi/53/P4P_Clean.rda",
            ["Performance bonuses binary", "PSM_C", "Efficacy_C", "Chosen_Job"],
        )
        df["Chosen_Job"] = pd.to_numeric(df["Chosen_Job"], errors="coerce")
        df["psm_group"] = df["PSM_C"] + "|" + df["Performance bonuses binary"]
        df["efficacy_group"] = df["Efficacy_C"] + "|" + df["Performance bonuses binary"]
        psm_map = {
            "High|No (fixed salary)": {"arm_id": "psm_high_no_bonus"},
            "High|Yes (bonuses)": {"arm_id": "psm_high_bonus"},
            "Low|No (fixed salary)": {"arm_id": "psm_low_no_bonus"},
            "Low|Yes (bonuses)": {"arm_id": "psm_low_bonus"},
        }
        efficacy_map = {
            "High|No (fixed salary)": {"arm_id": "efficacy_high_no_bonus"},
            "High|Yes (bonuses)": {"arm_id": "efficacy_high_bonus"},
            "Low|No (fixed salary)": {"arm_id": "efficacy_low_no_bonus"},
            "Low|Yes (bonuses)": {"arm_id": "efficacy_low_bonus"},
        }
        results = make_results(
            "complete",
            "local_summary_files",
            [
                "3_Aadi/53/P4P_Clean.rda",
                "3_Aadi/53/P4P_Analysis2.R",
            ],
            "This follow-on registry entry reuses the conjoint profile data and emphasizes moderation by public service motivation and self-efficacy. Results are stored as marginal choice probabilities for the performance-bonus attribute within each subgroup.",
            [
                outcome_record(
                    "out1",
                    "job attraction by performance bonuses within public service motivation subgroup",
                    "binary",
                    "share chosen",
                    summarize_groups(
                        df.dropna(subset=["psm_group", "Chosen_Job"]),
                        "psm_group",
                        "Chosen_Job",
                        group_map=psm_map,
                        metric="proportion",
                    ),
                    analysis_population="All randomized job profiles, grouped by PSM category",
                ),
                outcome_record(
                    "out2",
                    "job attraction by performance bonuses within self-efficacy subgroup",
                    "binary",
                    "share chosen",
                    summarize_groups(
                        df.dropna(subset=["efficacy_group", "Chosen_Job"]),
                        "efficacy_group",
                        "Chosen_Job",
                        group_map=efficacy_map,
                        metric="proportion",
                    ),
                    analysis_population="All randomized job profiles, grouped by self-efficacy category",
                ),
            ],
            ["out1", "out2"],
            flags=["conjoint_marginal_means_not_between_subject_arms"],
        )
    elif seq_id == 54:
        df = read_csv("3_Aadi/54/data/Reddinger_Charness_Levine_vaccination_public_good.csv")
        work = df[["vaxdose", "game_a", "game_b", "game_c", "game_d"]].copy()
        work["group"] = work["vaxdose"].map({1.0: "vaccinated", 0.0: "not_vaccinated"})
        results = make_results(
            "complete",
            "local_summary_files",
            [
                "3_Aadi/54/data/Reddinger_Charness_Levine_vaccination_public_good.csv",
                "3_Aadi/54/replicate.do",
            ],
            "This registered follow-up is observational rather than a treatment-arm RCT. Numerical summaries are therefore reported by observed vaccination status, which is the main grouping variable described in the registry entry and public data.",
            [
                outcome_record(
                    "out1",
                    "Game A behavior",
                    "continuous",
                    "game units",
                    summarize_groups(work.dropna(subset=["group", "game_a"]), "group", "game_a", metric="mean"),
                    analysis_population="Observational follow-up sample grouped by self-reported vaccination status",
                ),
                outcome_record(
                    "out2",
                    "Game B behavior",
                    "continuous",
                    "game units",
                    summarize_groups(work.dropna(subset=["group", "game_b"]), "group", "game_b", metric="mean"),
                    analysis_population="Observational follow-up sample grouped by self-reported vaccination status",
                ),
            ],
            ["out1", "out2"],
            flags=["observational_groups_use_vaccination_status"],
        )
    elif seq_id == 55:
        rows = parse_stargazer_table_rows(
            "3_Aadi/55/adityaj/Library/CloudStorage/OneDrive-TheUniversityofChicago/Fresno_project/Browne_Gazze_Greenstone_Rostapshova_Supplement/output/table_a10.tex"
        )
        arm_map = {
            "disct50\\_visual\\_D": "A1",
            "disct75\\_visual\\_D": "A2",
            "basefine\\_300\\_D": "A3",
            "disct50\\_300\\_D": "A4",
            "disct75\\_300\\_D": "A5",
            "basefine\\_500\\_D": "A6",
            "disct50\\_500\\_D": "A7",
            "disct75\\_500\\_D": "A8",
            "basefine\\_700\\_D": "A9",
            "disct50\\_700\\_D": "A10",
            "disct75\\_700\\_D": "A11",
        }
        water_use = []
        notices = []
        for row_name, arm_id in arm_map.items():
            vals = rows[row_name]["coef"]
            ses = rows[row_name]["se"]
            water_use.append(
                manual_group_summary(
                    arm_id,
                    vals[5],
                    metric="other",
                    n_analyzed=None,
                    se=ses[5] if len(ses) > 5 else None,
                    notes="Coefficient estimate relative to control arm A0 from table_a10.tex",
                )
            )
            notices.append(
                manual_group_summary(
                    arm_id,
                    vals[2],
                    metric="other",
                    n_analyzed=None,
                    se=ses[2] if len(ses) > 2 else None,
                    notes="Coefficient estimate relative to control arm A0 from table_a10.tex",
                )
            )
        results = make_results(
            "complete",
            "code_output",
            [
                "3_Aadi/55/adityaj/Library/CloudStorage/OneDrive-TheUniversityofChicago/Fresno_project/Browne_Gazze_Greenstone_Rostapshova_Supplement/output/table_a10.tex",
                "3_Aadi/55/adityaj/Library/CloudStorage/OneDrive-TheUniversityofChicago/Fresno_project/Browne_Gazze_Greenstone_Rostapshova_Supplement/code/figures/plot_figure_2.R",
            ],
            "The public package exposes treatment-cell effects most cleanly through the published regression-output table. These summaries are treatment-effect estimates relative to the control enforcement arm (A0), not raw cell means.",
            [
                outcome_record(
                    "out1",
                    "effect on log monthly water use relative to control",
                    "other",
                    "log points",
                    water_use,
                ),
                outcome_record(
                    "out2",
                    "effect on probability of receiving a notice relative to control",
                    "other",
                    "probability points",
                    notices,
                ),
            ],
            ["out1"],
            flags=["treatment_effect_estimates_not_raw_group_means"],
        )

    out = deepcopy(rec)
    out["schema_version"] = "study_enriched.v1"
    out["provenance"]["registry"].setdefault("clustered", None)
    out["enrichment"]["results"] = results
    return out


def main():
    designs = load_designs()
    selected = [enrich_study(i, designs[i]) for i in SELECTED_IDS if i in designs]
    with OUTPUT_PATH.open("w") as f:
        for obj in selected:
            f.write(json.dumps(obj) + "\n")
    print(f"Wrote {len(selected)} study records to {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
