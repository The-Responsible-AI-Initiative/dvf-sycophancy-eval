# analysis/

## `build_paper_tables.py`

Regenerates Table 1, Table 2, and Figure 1 to match `AIES_Camera_Ready.pdf`
exactly. Run with:

```
python analysis/build_paper_tables.py
```

Output goes to `results/tables/table1_raw_scores.{csv,tex}`,
`results/tables/table2_onset_turns.{csv,tex}`, and
`results/figures/figure1_dvf_scores.{pdf,png}`.

**Important caveat — this is a placeholder, not the real pipeline.** The
script reads `data/paper_reported_results.json`, which holds the *published*
Table 1 / Table 2 summary statistics (mean/SD per model x dimension x age,
and per-dimension turn onsets) transcribed directly from the paper. It is
**not** derived from the original per-conversation judge scores — those
have not yet been located in this repo (the experiments were run and real
output exists somewhere; it's being tracked down). Figure 1's `V(m,a)`
values are computed from those published cell means via the DVF formula in
`dvf/dvf.py` / `constants.py`; this has been verified to reproduce every
`V(m,a)` value cited in the paper's Discussion section exactly (e.g.
GPT-5.4-mini `V(a1)=0.212`, Gemini `V(a1)=0.292`, DeepSeek `V(a1)=0.423` and
`V(a2)=0.211`, Claude's range `[0.063, 0.094]`).

**When the real per-conversation data is found:** replace
`load_reported_data()` in `build_paper_tables.py` with a loader that
aggregates mean/SD directly from the real judge-scored conversations, and
delete `data/paper_reported_results.json`. The table/figure-building logic
below `load_reported_data()` doesn't need to change — it only needs real
per-cell `[mean, sd]` and onset-turn values in the same shape.

## Prior state

The analysis scripts that used to live here (`compute_dvf.py`,
`fill_gaps.py`, the old `tables.py`, `plot_trajectories.py`) were built
around an earlier, abandoned study design (SYCON Bench-sourced E/F, no S
dimension, a "GPT-4o (SYCON)" model) and have been moved to
`archive/legacy-sycon-pipeline/` — see `archive/README.md`.

Separately, a unified judge over `results/raw/` conversations (the paper
specifies Llama 3.3 70B with dimension-specific 4-question binary rubrics —
see the rubric text already in `judge/judge.py`, though it currently runs
on `llama3.1:8b`; `judge/emotional_judge.py` and `judge/epistemic_judge.py`
use different models/rubrics and are not consistent with each other or
with the paper's stated method) is still needed to regenerate these tables
from first principles once the real scored data is found or re-produced.
