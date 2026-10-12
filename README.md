# dvf-sycophancy-eval

Code and data for:

> **Frictionless Validation: Exploring AI Sycophancy and Developmental Risks**
> Junfeng Jiao, Abhejay Murali, Saleh Afroogh — Urban Information Lab, The University of Texas at Austin
> AIES 2026. Full text: [`AIES_Camera_Ready.pdf`](./AIES_Camera_Ready.pdf)

## What this is

We evaluate four frontier consumer LLMs (Claude Sonnet 4.6, GPT-5.4-mini,
Gemini 3.5 Flash, DeepSeek V4 Pro) for sycophancy toward simulated
adolescent users, using age-calibrated conversational probes across three
developmental configurations (early 7–9, middle 10–12, late 13–17) and
three sycophancy dimensions:

- **Epistemic (E)** — validating a factually flawed belief
- **Emotional (S)** — mirroring/amplifying emotional dependence
- **Framing (F)** — accepting a biased moral or causal framing

To formalize risk across developmental stages we introduce the
**Developmental Vulnerability Function (DVF)**:

```
V_t(m, a) = sum_s  alpha_s(a) * delta_s^adult * sigma_s(m, t)
```

where `sigma_s(m, t)` is a judge-assigned per-turn sycophancy score in
`[0, 1]`, `alpha_s(a)` is a theory-stipulated developmental amplification
factor, and `delta_s^adult` is an adult-effect calibration anchor from
Cheng et al. 2026a (*Science*). See `AIES_Camera_Ready.pdf` (Methodology)
for the full derivation and `dvf/dvf.py` / `constants.py` for the
implementation.

## Repository layout

```
agents.py              9 probe system prompts (3 ages x 3 dimensions)
runner.py              main conversation runner (+ per-model variants:
                        runner_deepseek.py, runner_gemini.py, runner_gpt.py)
rerun_deepseek_a1_F.py one-off targeted rerun script
judge/                 three LLM-judge scripts scoring conversations for
                        sycophancy (see judge/*.py headers for current
                        status — they are not yet mutually consistent;
                        see analysis/README.md)
dvf/                   DVF formula implementation (dvf.py), shared
                        constants (constants.py)
constants.py            root-level mirror of dvf/constants.py
analysis/               currently empty — see analysis/README.md for what's
                        needed to regenerate the paper's tables/figures
data/                  conversation data, scored results, calibration
                        artifacts, run logs
results/               raw/failed/invalid conversation outputs from
                        experimental runs
archive/               superseded code and data from earlier study
                        designs, kept for history (see archive/README.md)
frictionless-validation-aies2026-supplementary/
                        the AIES supplementary-materials bundle submitted
                        alongside the paper (probe prompts, raw
                        conversations, a standalone DVF script)
```

## Reproducing a run

Requirements: Python 3.11+, `pip install -r requirements.txt`, a local
[Ollama](https://ollama.com) instance for the judge models, and API keys
for the target models.

1. Copy `.env.example` (or create `.env`) with:
   ```
   ANTHROPIC_API_KEY=...
   OPENAI_API_KEY=...
   GOOGLE_API_KEY=...
   DEEPSEEK_API_KEY=...
   ```
2. Pull the Ollama models used by the judges/probes, e.g. `ollama pull llama3.1:8b`.
3. Run conversations: `python runner.py --dry-run` to preview the plan, then
   `python runner.py` for a real run. Output lands in `results/raw/`.
4. Score conversations: `python judge/judge.py results/raw/` (see
   `judge/*.py` headers for which judge is appropriate for which dimension
   — this is a known gap, not yet unified).
5. Regenerate Table 1 / Table 2 / Figure 1: `python analysis/build_paper_tables.py`.
   **As of now this reads published values transcribed from the paper
   (`data/paper_reported_results.json`), not the original per-conversation
   judge scores** — see `analysis/README.md` for the caveat and how to swap
   in real data once located.

## Known limitations

- `analysis/build_paper_tables.py` currently reproduces the paper's Table
  1/2/Figure 1 numbers from transcribed published values, not from
  re-aggregated per-conversation judge scores. The original scored data is
  being tracked down; see `analysis/README.md`.
- The judge scripts in `judge/` use different models and rubric formats
  from each other and have not been confirmed to reproduce the paper's
  published numbers from raw conversations. See the status note at the top
  of each file.
- Three raw conversations (Gemini 3.5 Flash, late developmental config,
  framing dimension) are missing from this repo.
- `archive/` holds an earlier, abandoned analysis design (SYCON
  Bench-sourced scoring, a 5th "GPT-4o" comparison model) — it is not used
  by, and does not match, the camera-ready paper.

## License

[MIT](./LICENSE)

## Citation

```bibtex
@inproceedings{jiao2026frictionless,
  title     = {Frictionless Validation: Exploring AI Sycophancy and Developmental Risks},
  author    = {Jiao, Junfeng and Murali, Abhejay and Afroogh, Saleh},
  booktitle = {Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society},
  year      = {2026}
}
```
