"""Recompute manuscript summaries from the supplied DTAA analysis archive."""

import argparse
from collections import Counter
import csv
import json
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact, norm


ERAS = ((2015, 2019), (2020, 2022), (2023, 2025))
FAMILIES = ("prediction", "causal")
STRICT_DISCHARGE_UNCERTAIN = {
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "discharge_disposition_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "payer_code",
}
REQUIRED_INPUTS = (
    "all_records.json",
    "screening_decisions_final_v2.json",
    "review_extraction_paperlevel.csv",
    "variable_audit_manual.json",
    "variable_registry.csv",
    "crossdomain_audit_matrix_v2.csv",
    "simulation_bias_results.json",
    "earlydrain_reproduction_results.json",
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def reporting_proportion(papers):
    stated = sum(p["dp_stated_adjudicated"] == "yes" for p in papers)
    uncertain = sum(p["dp_stated_adjudicated"] == "unclear" for p in papers)
    low, high = binomtest(stated, len(papers)).proportion_ci(method="exact")
    return {
        "stated": stated,
        "uncertain": uncertain,
        "total": len(papers),
        "proportion": stated / len(papers),
        "ci95": [low, high],
    }


def literature_summaries(archive):
    records = read_json(archive / "all_records.json")
    screening = read_json(archive / "screening_decisions_final_v2.json")
    extracted = read_csv(archive / "review_extraction_paperlevel.csv")
    included = [p for p in extracted if p["population_ok"] != "no"]
    included_ids = {p["pmid"] for p in included}
    if len(included_ids) != len(included):
        raise ValueError("The included literature archive contains duplicate PMIDs.")
    by_publication_year = {}
    for paper in included:
        year = int(paper["year"])
        if year < 2015 or year > 2026:
            raise ValueError(f"Unexpected issue year in the frozen audit archive: {year}")
        # PubMed verification confirmed that the 2026-issue records in this archive
        # were all published online in 2025.
        by_publication_year[paper["pmid"]] = min(year, 2025)

    reporting = {
        "overall": reporting_proportion(included),
        "by_family": {
            family: reporting_proportion([p for p in included if p["model_family"] == family])
            for family in FAMILIES
        },
        "by_era": {},
        "by_era_and_family": {},
    }
    for start, end in ERAS:
        label = f"{start}-{end}"
        papers = [
            p for p in included if start <= by_publication_year[p["pmid"]] <= end
        ]
        reporting["by_era"][label] = reporting_proportion(papers)
        reporting["by_era_and_family"][label] = {
            family: reporting_proportion([p for p in papers if p["model_family"] == family])
            for family in FAMILIES
        }

    scores = np.arange(1, len(ERAS) + 1)
    era_totals = np.array([p["total"] for p in reporting["by_era"].values()])
    era_stated = np.array([p["stated"] for p in reporting["by_era"].values()])
    centered_scores = scores - np.average(scores, weights=era_totals)
    p_all = reporting["overall"]["proportion"]
    z = (centered_scores @ era_stated) / sqrt(
        p_all * (1 - p_all) * (era_totals @ (centered_scores ** 2))
    )
    reporting["era_trend_p_two_sided"] = float(2 * norm.sf(abs(z)))
    family_counts = [
        [row["stated"], row["total"] - row["stated"]] for row in reporting["by_family"].values()
    ]
    reporting["family_fisher_p_two_sided"] = float(fisher_exact(family_counts).pvalue)

    manual = read_json(archive / "variable_audit_manual.json")
    analysed = [p for p in manual if p["n_vars"] and p["pmid"] in included_ids]
    counts = np.array([[p[k] for k in ("n_T0", "n_T1", "n_POST", "n_U")] for p in analysed])
    totals = np.array([p["n_vars"] for p in analysed])
    if not np.array_equal(counts.sum(axis=1), totals):
        raise ValueError("The predictor classes do not sum to the model totals.")
    fractions = counts[:, 1:] / totals[:, None]
    combined = fractions.sum(axis=1)
    bootstrap = np.random.default_rng(20261126).choice(combined, (10_000, len(combined)))
    ci = np.quantile(bootstrap.mean(axis=1), [.025, .975]).tolist()
    return {
        "selection": {
            "deduplicated": len(records),
            "eligibility_screening": len(screening),
            "initially_eligible": sum(p["decision"] == "include" for p in screening),
            "included": len(included),
        },
        "reporting": reporting,
        "predictor_timing": {
            "papers": len(analysed),
            "variables": int(totals.sum()),
            "all_variables_available": int(np.count_nonzero(combined == 0)),
            "mean_post_decision": float(fractions[:, :2].sum(axis=1).mean()),
            "mean_uncertain": float(fractions[:, 2].mean()),
            "mean_post_decision_or_uncertain": float(combined.mean()),
            "median_post_decision_or_uncertain": float(np.median(combined)),
            "paper_bootstrap_ci95": ci,
        },
    }


def uci_anchor_summaries(archive):
    rows = read_csv(archive / "crossdomain_audit_matrix_v2.csv")
    summaries = {}
    for anchor in ("A1", "A2", "A2b"):
        selected = [
            r for r in rows
            if r["anchor"] == anchor and r["causal_role"] not in {"identifier", "outcome"}
        ]
        fields = {r["raw_column"] for r in selected}
        if len(selected) != 47 or len(fields) != 47:
            raise ValueError(f"{anchor} does not contain the 47 distinct candidate features.")
        if anchor == "A2" and not STRICT_DISCHARGE_UNCERTAIN <= fields:
            raise ValueError("The discharge audit is missing one or more timing-uncertain fields.")

        timing_counts = Counter()
        clinically_available = shared_at_decision = 0
        for row in selected:
            timing, accessibility = row["T_t0"], row["C_clinical"]
            if anchor == "A2" and row["raw_column"] in STRICT_DISCHARGE_UNCERTAIN:
                timing = accessibility = "U"
            if timing not in {"0", "1", "U"}:
                raise ValueError(f"Unexpected candidate timing class at {anchor}: {timing}")
            timing_counts[timing] += 1
            clinically_available += timing == "1" and accessibility == "1"
            shared_at_decision += timing == "1" and row["R_shared"] == "1"
        summaries[anchor] = {
            "features": len(selected),
            "timing_counts": {key: timing_counts[key] for key in ("1", "0", "U")},
            "DAI_C": clinically_available / len(selected),
            "DAI_D": shared_at_decision / len(selected),
            "PAVR": timing_counts["0"] / len(selected),
            "PUR": timing_counts["U"] / len(selected),
        }
    return summaries


def registry_summaries(archive):
    rows = read_csv(archive / "variable_registry.csv")
    result = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        entries = [row for row in rows if row["dataset"] == dataset]
        result[dataset] = {
            "entries": len(entries),
            "temporal_availability": dict(Counter(r["T_t0"] for r in entries)),
            "shared_status": dict(Counter(r["R_shared"] for r in entries)),
            "clinical_accessibility": dict(Counter(r["C_clinical"] for r in entries)),
        }
    return result


def simulation_summaries(archive):
    simulation = read_json(archive / "simulation_bias_results.json")
    conditions = simulation["cells"]
    mechanisms = {"M": "treatment_related", "C": "collider", "O": "outcome_descendant", "N": "noise"}
    validation_slice = [
        {
            "mechanism": mechanisms[next(iter(r["post_counts"]))],
            "true_risk_difference": r["truth"],
            "post_adjustment": r["post"],
        }
        for r in conditions
        if r["randomized"] and r["n"] == 287
        and len(r["post_counts"]) == 1 and sum(r["post_counts"].values()) == 8
    ]
    mixed, = [r for r in conditions if r["post_counts"] == {"M": 5, "C": 2, "N": 1}]
    base_randomised = sum(bool(r["randomized"]) for r in conditions) - 1
    return {
        "conditions": len(conditions),
        "randomised_grid_conditions": base_randomised,
        "confounded_conditions": len(conditions) - base_randomised - 1,
        "replicates_per_condition": simulation["meta"]["reps"],
        "eight_variable_slice": validation_slice,
        "mixed_scenario": mixed,
    }


def earlydrain_contrast_summaries(archive):
    data = read_json(archive / "earlydrain_reproduction_results.json")
    result = {}
    for row in data["rct_contrast"]:
        a, n1 = row["events_LD"], row["n_LD"]
        b, n0 = row["events_NoLD"], row["n_NoLD"]
        risk_ratio = (a / n1) / (b / n0)
        se = sqrt(1 / a - 1 / n1 + 1 / b - 1 / n0)
        width = norm.ppf(.975) * se
        result[row["outcome"]] = {
            "events_ld": a,
            "total_ld": n1,
            "events_no_ld": b,
            "total_no_ld": n0,
            "risk_ratio": risk_ratio,
            "wald_ci95": [exp(log(risk_ratio) - width), exp(log(risk_ratio) + width)],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Directory with the supplied analysis inputs.")
    args = parser.parse_args()
    missing = [name for name in REQUIRED_INPUTS if not (args.archive / name).is_file()]
    if missing:
        parser.error("Missing analysis inputs: " + ", ".join(missing))
    result = {
        "literature_audit": literature_summaries(args.archive),
        "uci_anchors": uci_anchor_summaries(args.archive),
        "aneurysm_registries": registry_summaries(args.archive),
        "simulation_archive": simulation_summaries(args.archive),
        "earlydrain_randomised_contrasts": earlydrain_contrast_summaries(args.archive),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
