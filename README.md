# DTAA analysis reproduction

Analysis code for *Auditing predictor availability at the point of clinical decision making*.

## Run

Use Python 3.11 or later.

```sh
python -m pip install -r requirements.txt
python test_benchmark.py
python reproduce_summaries.py --archive /path/to/analysis-archive
```

The benchmark command runs seven checks using constructed examples. The summary command prints aggregate JSON from a separately supplied archive.

## Inputs and coverage

| Archive inputs | Reproduced summary |
| --- | --- |
| all_records.json, screening_decisions_final_v2.json, review_extraction_paperlevel.csv and variable_audit_manual.json | Literature screening counts, combined reporting outcome, exact binomial intervals, era trend and model-family tests, and the paper-level timing bootstrap |
| variable_registry.csv and crossdomain_audit_matrix_v2.csv | Per-layer registry counts and the three UCI decision-anchor summaries, including the stricter A2 timing rule |
| simulation_bias_results.json | Counts and summary statistics of archived simulation scenarios |
| earlydrain_reproduction_results.json | Unadjusted risk ratios and Wald intervals reconstructed from archived EARLYDRAIN event counts |

The summary script starts from archived extractions, classifications or aggregate results. It does not regenerate the simulation replicates or refit the EARLYDRAIN DML, coiling prediction or Familial sensitivity models. The archive is separate from this repository.

For era stratification, the 11 records assigned to a 2026 issue were included in the 2023–2025 group after their 2025 online publication dates were verified in PubMed. All other included records use their archived year.
