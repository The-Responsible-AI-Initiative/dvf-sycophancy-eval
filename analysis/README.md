# analysis/

## `build_paper_tables.py`

Regenerates Table 1, Table 2, and Figure 1 from the AIES 2026 camera-ready
paper. Run with:

```
python analysis/build_paper_tables.py
```

Output goes to `results/tables/table1_raw_scores.{csv,tex}`,
`results/tables/table2_onset_turns.{csv,tex}`, and
`results/figures/figure1_dvf_scores.{pdf,png}`.

It reads `data/paper_reported_results.json` — the paper's reported Table 1
(mean/SD raw sycophancy score per model x dimension x age) and Table 2
(per-dimension turn onset) — and computes the cumulative `V(m,a)` DVF
scores from those per-cell means using `dvf/dvf.py` / `constants.py`. This
is the same formula the paper itself uses to go from raw sigma scores to
`V(m,a)`; recomputing it here reproduces every `V(m,a)` value cited in the
paper's Discussion section exactly.

## Running the framework on your own data

The DVF methodology generalizes beyond this paper's four models. To apply
it to your own experiments:

1. Run conversations through age-calibrated probes (see `agents.py` /
   `runner.py` for the pattern, or write your own).
2. Score them with a judge over the three dimensions (see `judge/` for
   worked examples of the rubric structure).
3. Aggregate into per-cell `[mean, sd]` values, in the same shape as
   `table1_raw_sigma` in `data/paper_reported_results.json`:
   `{model: {age_config: {dimension: [mean, sd]}}}`.
4. Point `load_reported_data()` in `build_paper_tables.py` at your file (or
   pass a different path) and rerun — Table 1, Table 2, and Figure 1
   regenerate using the same `ALPHA` / `DELTA_ADULT` / `THETA` weights.

`ALPHA`, `DELTA_ADULT`, `THETA`, and `TAU` in `constants.py` are the
theory-stipulated developmental weights and adult calibration anchors from
the paper; adjust them if you're calibrating the framework to a different
population or set of adult effect sizes.

## `archive/legacy-sycon-pipeline/`

An earlier analysis design (SYCON Bench-sourced E/F dimensions, a
"GPT-4o (SYCON)" comparison model) that predates the final three-dimension
probe methodology. Kept for history; see `archive/README.md`.
