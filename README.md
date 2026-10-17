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
judge/                 LLM-judge implementations scoring conversations for
                        sycophancy across the three dimensions — three
                        worked examples with different model/rubric choices
                        you can pick from or adapt (see judge/*.py headers)
dvf/                   DVF formula implementation (dvf.py), shared
                        constants (constants.py)
constants.py            root-level mirror of dvf/constants.py
analysis/               build_paper_tables.py regenerates Table 1, Table 2,
                        and Figure 1 from the paper's reported results, and
                        doubles as the template for running the framework
                        on your own data — see analysis/README.md
data/                  conversation data, scored results, calibration
                        artifacts, run logs, and the paper's reported
                        results (paper_reported_results.json)
results/               raw/failed/invalid conversation outputs from
                        experimental runs
archive/               superseded code and data from earlier study
                        designs, kept for history (see archive/README.md)
frictionless-validation-aies2026-supplementary/
                        the AIES supplementary-materials bundle submitted
                        alongside the paper (probe prompts, raw
                        conversations, a standalone DVF script)
```

## Regenerating the paper's results

```
pip install -r requirements.txt
python analysis/build_paper_tables.py
```

This regenerates Table 1, Table 2, and Figure 1 from the paper's reported
results (`data/paper_reported_results.json`) using the DVF formula in
`dvf/dvf.py` / `constants.py`. Output lands in `results/tables/` and
`results/figures/`.

## Running the framework on your own experiments

Requirements: Python 3.11+, a local [Ollama](https://ollama.com) instance
for the judge/probe models, and API keys for whichever target models you
want to evaluate.

1. Create `.env` with the keys for the providers you're targeting, e.g.:
   ```
   ANTHROPIC_API_KEY=...
   OPENAI_API_KEY=...
   GOOGLE_API_KEY=...
   DEEPSEEK_API_KEY=...
   ```
2. Pull the Ollama models used by the probes/judges, e.g. `ollama pull llama3.1:8b`.
3. Run conversations: `python runner.py --dry-run` to preview the plan, then
   `python runner.py` for a real run. Output lands in `results/raw/`. Adapt
   `agents.py` (probe prompts) and `TARGET_MODELS` in `runner.py` to your
   own models/scenarios.
4. Score conversations with a judge from `judge/` — three worked examples
   are included with different model/rubric choices; pick one or adapt it
   to your rubric.
5. Combine your scores into the same format as data/paper_reported_results.json, then configure analysis/build_paper_tables.py to use your file so it can regenerate Table 1, Table 2, and Figure 1 for your models. Refer to analysis/README.md for the complete step-by-step guide.

## Notes

- `judge/` contains three independent judge implementations (different
  judge models, different rubric formats) rather than one canonical
  pipeline — each is a usable reference you can adapt, not a single
  required path.
- `archive/` holds an earlier analysis design (SYCON Bench-sourced scoring,
  a 5th "GPT-4o" comparison model) that predates the paper's final
  three-dimension probe methodology. Kept for history; see `archive/README.md`.

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
