"""
Regenerates Table 1, Table 2, and Figure 1 from AIES_Camera_Ready.pdf.

DATA SOURCE CAVEAT
-------------------
This reads data/paper_reported_results.json, which holds the *published*
Table 1 / Table 2 summary statistics transcribed directly from the paper --
NOT the original per-conversation judge scores (those haven't been located
in this repo yet; see analysis/README.md). V(m, a) for Table 1 cumulative
scores / Figure 1 is computed from those published per-cell means using the
DVF formula in dvf/dvf.py, which is how the paper itself derives V(m,a) from
sigma_bar_s (verified: recomputing V(m, a1) for all four models from Table 1
here reproduces every V(m,a) value cited in the paper's Discussion section
to 3 decimal places). Table 2's turn-onset values cannot be derived this way
(they need per-turn trajectories, not cell means) and are transcribed as-is.

Once the real per-conversation scores are located, replace the loader below
(`load_reported_data`) with one that aggregates from the real data, and
delete data/paper_reported_results.json.

Usage
-----
    python analysis/build_paper_tables.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from constants import ALPHA, DELTA_ADULT, THETA

DATA_FILE   = os.path.join(_ROOT, "data", "paper_reported_results.json")
TABLES_DIR  = os.path.join(_ROOT, "results", "tables")
FIGURES_DIR = os.path.join(_ROOT, "results", "figures")

MODEL_ORDER = ["claude_sonnet", "gpt55", "gemini_flash", "deepseek"]
AGE_ORDER   = ["a1", "a2", "a3"]
DIM_ORDER   = ["S", "F", "E"]


def load_reported_data(path: str = DATA_FILE) -> dict:
    with open(path) as f:
        return json.load(f)


def vulnerability_weight(age: str, dim: str) -> float:
    """W_s(a) = alpha_s(a) * delta_s^adult, per dvf/dvf.py / constants.py."""
    return ALPHA[age][dim] * DELTA_ADULT[dim]


def compute_v(data: dict, model: str, age: str) -> float:
    """
    V(m, a) = sum_s W_s(a) * sigma_bar_s(m, a), using Table 1's published
    per-cell mean raw sigma scores. Matches dvf.compute_vt applied to a
    single-cell mean rather than a full per-turn trajectory.
    """
    cell = data["table1_raw_sigma"][model][age]
    return sum(
        vulnerability_weight(age, dim) * cell[dim][0]
        for dim in ("E", "S", "F")
    )


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------

def build_table1(data: dict) -> str:
    models = data["models"]
    ages   = data["ages"]
    lines  = ["Model,D,Early,Middle,Late"]
    for m in MODEL_ORDER:
        for dim in DIM_ORDER:
            row = [models[m] if dim == "S" else "", dim]
            for a in AGE_ORDER:
                mean, sd = data["table1_raw_sigma"][m][a][dim]
                row.append(f"{mean:.3f} ({sd:.3f})")
            lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def build_table1_latex(data: dict) -> str:
    models = data["models"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & D & Early & Middle & Late \\",
        r"\midrule",
    ]
    for m in MODEL_ORDER:
        label = models[m]
        for i, dim in enumerate(DIM_ORDER):
            cells = []
            for a in AGE_ORDER:
                mean, sd = data["table1_raw_sigma"][m][a][dim]
                cells.append(f"{mean:.3f} ({sd:.3f})")
            name = label if i == 0 else ""
            lines.append(f"{name} & {dim} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Mean raw sycophancy score $\bar{\sigma}_s$ per model, "
              r"dimension, and age configuration ($n{=}3$ runs, $T{=}8$ turns). "
              r"Standard deviations in parentheses.}",
              r"\label{tab:raw_scores}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 2
# ---------------------------------------------------------------------------

def build_table2(data: dict) -> str:
    models = data["models"]
    lines  = ["Model,Early S,Early F,Middle S,Middle F,Late S,Late F"]
    for m in MODEL_ORDER:
        row = [models[m]]
        for a in AGE_ORDER:
            for dim in ("S", "F"):
                v = data["table2_onset_turn"][m][a][dim]
                row.append(str(v) if v is not None else "-")
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def build_table2_latex(data: dict) -> str:
    models = data["models"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Early} & \multicolumn{2}{c}{Middle} & \multicolumn{2}{c}{Late} \\",
        r"Model & S & F & S & F & S & F \\",
        r"\midrule",
    ]
    for m in MODEL_ORDER:
        cells = []
        for a in AGE_ORDER:
            for dim in ("S", "F"):
                v = data["table2_onset_turn"][m][a][dim]
                cells.append(str(v) if v is not None else "--")
        lines.append(f"{models[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Dimension-specific raw-score onset $t_s^\sigma$ for "
              r"the emotional (S) and framing (F) dimensions by model and "
              r"age configuration. A dash indicates the raw sycophancy "
              r"score did not reach $\tau{=}0.20$ within $T{=}8$ turns.}",
              r"\label{tab:onset}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

def build_figure1(data: dict, out_dir: str = FIGURES_DIR) -> dict:
    models = data["models"]
    v = {m: {a: compute_v(data, m, a) for a in AGE_ORDER} for m in MODEL_ORDER}

    age_labels = [data["ages"][a] for a in AGE_ORDER]
    x = np.arange(len(AGE_ORDER))
    n_models = len(MODEL_ORDER)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#DDDDDD", "#999999", "#555555", "#111111"]
    for i, m in enumerate(MODEL_ORDER):
        heights = [v[m][a] for a in AGE_ORDER]
        ax.bar(x + i * width - 0.4 + width / 2, heights, width,
               label=models[m], color=colors[i % len(colors)],
               edgecolor="black", linewidth=0.5)

    ax.axhline(THETA, color="red", linestyle="--", linewidth=1,
               label=f"θ = {THETA:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(age_labels)
    ax.set_ylabel("V(m, a)")
    ax.set_ylim(0, 0.45)
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    ax.set_title("DVF scores by age configuration")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "figure1_dvf_scores.pdf")
    png_path = os.path.join(out_dir, "figure1_dvf_scores.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return v


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    data = load_reported_data()

    os.makedirs(TABLES_DIR, exist_ok=True)

    t1_csv = build_table1(data)
    with open(os.path.join(TABLES_DIR, "table1_raw_scores.csv"), "w") as f:
        f.write(t1_csv)
    with open(os.path.join(TABLES_DIR, "table1_raw_scores.tex"), "w") as f:
        f.write(build_table1_latex(data))
    print("Table 1 (raw sigma scores):\n" + t1_csv)

    t2_csv = build_table2(data)
    with open(os.path.join(TABLES_DIR, "table2_onset_turns.csv"), "w") as f:
        f.write(t2_csv)
    with open(os.path.join(TABLES_DIR, "table2_onset_turns.tex"), "w") as f:
        f.write(build_table2_latex(data))
    print("Table 2 (onset turns):\n" + t2_csv)

    v = build_figure1(data)
    print("Figure 1 (V(m,a)):")
    for m in MODEL_ORDER:
        label = data["models"][m]
        vals = "  ".join(f"{data['ages'][a]}={v[m][a]:.3f}" for a in AGE_ORDER)
        flag = " *** crosses theta" if any(v[m][a] >= THETA for a in AGE_ORDER) else ""
        print(f"  {label:<18} {vals}{flag}")
    print(f"\nWritten to {TABLES_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    run()
