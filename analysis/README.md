# analysis/

Empty by design. The analysis scripts that used to live here
(`compute_dvf.py`, `fill_gaps.py`, `tables.py`, `plot_trajectories.py`) were
built around an earlier, abandoned study design (SYCON Bench-sourced E/F,
no S dimension, a "GPT-4o (SYCON)" model) and have been moved to
`archive/legacy-sycon-pipeline/` — see `archive/README.md`.

There is currently no script in this repo that reproduces the camera-ready
paper's Table 1, Table 2, or Figure 1 from `results/raw/` conversations.
Building one requires:

1. A single, consistent judge over all three dimensions (E, S, F) — the
   paper specifies Llama 3.3 70B with dimension-specific 4-question binary
   rubrics (see the rubric text already in `judge/judge.py`, which has the
   right structure but currently runs on `llama3.1:8b`; `judge/emotional_judge.py`
   and `judge/epistemic_judge.py` use different models/rubrics entirely and
   are not consistent with each other or with the paper's stated method).
2. An aggregator over the judged output that computes, per the paper's
   definitions in `constants.py` / `dvf/dvf.py`:
   - Table 1: mean +/- SD raw `sigma_s` by model x dimension x age
     (n=3 runs per cell).
   - Table 2: dimension-specific turn onset `t^sigma_s = min{t : sigma_s(m,t) >= TAU}`
     for S and F.
   - Figure 1: cumulative `V(m,a)` per model x age, summed over E/S/F using
     `ALPHA` x `DELTA_ADULT`, against the `THETA` reference line.

As of this cleanup, the repo's existing scored-data files
(`data/emotional_results.json`, `data/epistemic_scored_results.txt`) do not
reproduce the paper's published numbers (wrong n, different rubric,
different values), and three raw conversations (Gemini 3.5 Flash, late
config, framing dimension) are missing from both `results/raw/` and the
supplementary bundle. Re-scoring and filling that gap is required before
this directory can regenerate the paper's exact results.
