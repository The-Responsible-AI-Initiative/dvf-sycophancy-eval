"""
Generates results tables as CSV and LaTeX.

Table 1  V(m, a) — cumulative DVF score per model per age config
Table 2  t*       — collapse point per model per age config
Table 3  σ_s      — mean per-dimension raw sycophancy score
                    (sigma_E, sigma_S, sigma_F) per model

Output
------
results/tables/table1_dvf_scores.csv
results/tables/table1_dvf_scores.tex
results/tables/table2_collapse_points.csv
results/tables/table2_collapse_points.tex
results/tables/table3_dimension_scores.csv
results/tables/table3_dimension_scores.tex

Usage
-----
    python analysis/tables.py
"""

import json
import os
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS    = os.path.join(_ROOT, "data", "dvf_results.json")
TABLES_DIR = os.path.join(_ROOT, "results", "tables")

# ---------------------------------------------------------------------------
# Display constants — must match paper exactly
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "gpt4o":         "GPT-4o (SYCON)",
    "gpt55":         "GPT-5.4 mini",
    "claude_sonnet": "Claude 4.6 Sonnet",
    "gemini_flash":  "Gemini 3.5 Flash",
    "deepseek":      "DeepSeek-v4 Pro",
}

AGE_LABELS = {
    "a1": "Early (7–9)",
    "a2": "Middle (10–12)",
    "a3": "Late (13–17)",
}

MODEL_ORDER = ("gpt4o", "gpt55", "claude_sonnet", "gemini_flash", "deepseek")
AGE_ORDER   = ("a1", "a2", "a3")
DIM_ORDER   = ("E", "S", "F")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[dict]:
    """
    Loads dvf_results.json.

    Args:
        path: path to dvf_results.json

    Returns:
        list of conversation result dicts
    """
    with open(path) as f:
        data = json.load(f)
    return data["conversations"]


def atomic_write_csv(path: str, df: pd.DataFrame) -> None:
    """
    Writes a DataFrame to CSV atomically.

    Args:
        path: destination path
        df:   DataFrame to write
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dirpath = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dirpath, suffix=".tmp", delete=False
    ) as tmp:
        df.to_csv(tmp, index=True)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def atomic_write_text(path: str, content: str) -> None:
    """
    Writes a text string to path atomically.

    Args:
        path:    destination path
        content: string content
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dirpath = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dirpath, suffix=".tmp", delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def pivot_table(
    data: dict[tuple, float | None],
    models: list[str],
    ages: list[str],
    fmt: str = "{:.3f}",
    none_str: str = "—",
) -> pd.DataFrame:
    """
    Builds a (model × age) DataFrame from a (model, age) → value dict.

    Args:
        data:     dict keyed by (model, age) → scalar or None
        models:   ordered list of model keys
        ages:     ordered list of age config keys
        fmt:      format string for numeric values
        none_str: string to display for None values

    Returns:
        DataFrame with model rows and age columns
    """
    rows = {}
    for m in models:
        label = MODEL_LABELS.get(m, m)
        row = {}
        for a in ages:
            v = data.get((m, a))
            row[AGE_LABELS.get(a, a)] = (
                fmt.format(v) if v is not None else none_str
            )
        rows[label] = row
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Table 1 — DVF scores V(m, a)
# ---------------------------------------------------------------------------

def build_table1(conversations: list[dict]) -> pd.DataFrame:
    """
    Computes V(m, a) per (model, age_config, dimension) with equal E/F weighting.

    E records (debate + false_presupposition) are averaged within the E
    dimension first; F records (ethical) are taken directly. This prevents
    the implicit 2:1 E-to-F weighting that arises when three records are
    averaged naively. Results are reported in separate E and F columns per
    age configuration.

    Args:
        conversations: list of dvf_results conversation dicts

    Returns:
        DataFrame with model rows and (age, dim) column pairs:
        Early E, Early F, Middle E, Middle F, Late E, Late F
    """
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for conv in conversations:
        key = (conv["model"], conv["age_config"], conv["scenario_type"])
        buckets[key].append(conv["dvf_score"])

    models = [m for m in MODEL_ORDER
              if any(k[0] == m for k in buckets)]
    ages   = [a for a in AGE_ORDER
              if any(k[1] == a for k in buckets)]

    rows = {}
    for m in models:
        label = MODEL_LABELS.get(m, m)
        row = {}
        for a in ages:
            age_label = AGE_LABELS.get(a, a)
            for dim in ("E", "F"):
                scores = buckets.get((m, a, dim), [])
                row[f"{age_label} {dim}"] = (
                    f"{float(np.mean(scores)):.3f}" if scores else "—"
                )
        rows[label] = row

    return pd.DataFrame(rows).T


def table1_latex(df: pd.DataFrame) -> str:
    """
    Renders Table 1 as a LaTeX booktabs table string.

    Uses \\multicolumn spanning headers for each age group, with E and F
    sub-columns underneath.

    Args:
        df: output of build_table1()

    Returns:
        LaTeX table string
    """
    age_labels  = [AGE_LABELS[a] for a in AGE_ORDER
                   if any(f"{AGE_LABELS[a]} E" in c for c in df.columns)]
    n_ages      = len(age_labels)
    col_spec    = "l" + "rr" * n_ages

    # Row 1: age group spans
    span_parts  = [r"\multicolumn{1}{l}{Model}"]
    cmidrules   = []
    for i, lbl in enumerate(age_labels):
        col_start = 2 + i * 2          # 1-indexed, after Model column
        col_end   = col_start + 1
        # escape em-dashes for LaTeX
        lbl_tex = lbl.replace("–", "--")
        span_parts.append(
            rf"\multicolumn{{2}}{{c}}{{{lbl_tex}}}"
        )
        cmidrules.append(
            rf"\cmidrule(lr){{{col_start}-{col_end}}}"
        )
    span_row = " & ".join(span_parts) + r" \\"

    # Row 2: E / F sub-headers
    sub_parts = [""] + ["E & F"] * n_ages
    sub_row   = " & ".join(sub_parts) + r" \\"

    data_rows = []
    for idx, row in df.iterrows():
        data_rows.append(
            " & ".join([str(idx)] + list(row.values)) + r" \\"
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Developmental Vulnerability Score $V(m,a)$ "
        r"per model per adolescent age configuration, reported "
        r"separately for the Epistemic (E) and Framing (F) dimensions. "
        r"E scores average the debate and false-presupposition settings; "
        r"F scores reflect the ethical setting. "
        r"Higher scores indicate greater harm risk.}",
        r"\label{tab:dvf_scores}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        span_row,
        " ".join(cmidrules),
        sub_row,
        r"\midrule",
        *data_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2 — Collapse points t*
# ---------------------------------------------------------------------------

def build_table2(conversations: list[dict]) -> pd.DataFrame:
    """
    Computes modal collapse point t* per (model, age_config) cell.

    Where multiple conversations contribute to the same cell,
    the mean collapse turn is reported (None if never collapsed).

    Args:
        conversations: list of dvf_results conversation dicts

    Returns:
        DataFrame with model rows, age config columns
    """
    buckets: dict[tuple, list] = defaultdict(list)
    for conv in conversations:
        key = (conv["model"], conv["age_config"])
        buckets[key].append(conv["collapse_point"])

    aggregated = {}
    for k, vals in buckets.items():
        non_null = [v for v in vals if v is not None]
        aggregated[k] = (
            float(np.mean(non_null)) if non_null else None
        )

    models = [m for m in MODEL_ORDER
              if any(m == k[0] for k in aggregated)]
    ages   = [a for a in AGE_ORDER
              if any(a == k[1] for k in aggregated)]

    return pivot_table(
        aggregated, models, ages,
        fmt="{:.1f}", none_str="None",
    )


def table2_latex(df: pd.DataFrame) -> str:
    """
    Renders Table 2 as a LaTeX booktabs table string.

    Args:
        df: output of build_table2()

    Returns:
        LaTeX table string
    """
    col_spec = "l" + "r" * len(df.columns)
    header   = " & ".join(["Model"] + list(df.columns)) + r" \\"
    rows     = []
    for idx, row in df.iterrows():
        rows.append(" & ".join([str(idx)] + list(row.values)) + r" \\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textit{Supplementary.} "
        r"Mean collapse turn $t^*$ per model per age configuration. "
        r"$t^*$ is the first turn at which $V_t(m,a) \geq \theta$. "
        r"``None'' indicates no collapse detected. "
        r"Because collapse is determined by the sigma trajectory "
        r"from SYCON Bench -- which carries no developmental "
        r"parameterisation -- $t^*$ is invariant across age "
        r"configurations for a given model. "
        r"Developmental differentiation of collapse points "
        r"requires age-parameterised conversations and "
        r"is a direction for future work.}",
        r"\label{tab:collapse_points}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3 — Dimensional breakdown σ_s
# ---------------------------------------------------------------------------

def build_table3(conversations: list[dict]) -> pd.DataFrame:
    """
    Computes mean raw sycophancy score sigma_s per model per dimension.

    Averaged across all turns and all conversations for each model,
    giving a model-level dimensional fingerprint independent of alpha.

    Args:
        conversations: list of dvf_results conversation dicts

    Returns:
        DataFrame with model rows, dimension columns (E, S, F)
    """
    dim_label = {"E": "Epistemic (E)", "S": "Emotional (S)",
                 "F": "Framing (F)"}
    buckets: dict[tuple, list[float]] = defaultdict(list)

    for conv in conversations:
        for ts in conv["turn_scores"]:
            for dim in DIM_ORDER:
                key = (conv["model"], dim)
                buckets[key].append(float(ts[dim]))

    aggregated = {k: float(np.mean(v)) for k, v in buckets.items()}

    models = [m for m in MODEL_ORDER
              if any(m == k[0] for k in aggregated)]

    rows = {}
    for m in models:
        label = MODEL_LABELS.get(m, m)
        row   = {}
        for d in DIM_ORDER:
            v = aggregated.get((m, d))
            row[dim_label[d]] = f"{v:.3f}" if v is not None else "—"
        rows[label] = row

    return pd.DataFrame(rows).T


def table3_latex(df: pd.DataFrame) -> str:
    """
    Renders Table 3 as a LaTeX booktabs table string.

    Args:
        df: output of build_table3()

    Returns:
        LaTeX table string
    """
    col_spec = "l" + "r" * len(df.columns)
    header   = " & ".join(["Model"] + list(df.columns)) + r" \\"
    rows     = []
    for idx, row in df.iterrows():
        rows.append(" & ".join([str(idx)] + list(row.values)) + r" \\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean raw sycophancy score $\bar{\sigma}_s$ "
        r"per model per dimension, averaged across all turns "
        r"and conversations. Scores are in $[0,1]$. "
        r"Epistemic (E) scores derive from debate and "
        r"false-presupposition settings; Framing (F) scores "
        r"from the ethical setting (both from SYCON Bench). "
        r"Emotional (S) scores require purpose-built evaluation "
        r"and are not reported here; S${}=0$ reflects the "
        r"absence of an emotional dimension in SYCON Bench, "
        r"not a substantive finding.}",
        r"\label{tab:dimension_scores}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    results_path: str = RESULTS,
    tables_dir:   str = TABLES_DIR,
) -> None:
    """
    Generates all three results tables and saves as CSV and LaTeX.

    Args:
        results_path: path to dvf_results.json
        tables_dir:   directory where table files are saved
    """
    print("Loading DVF results...")
    conversations = load_results(results_path)

    # Table 1
    print("Building Table 1 (DVF scores)...")
    t1 = build_table1(conversations)
    atomic_write_csv(
        os.path.join(tables_dir, "table1_dvf_scores.csv"), t1
    )
    atomic_write_text(
        os.path.join(tables_dir, "table1_dvf_scores.tex"),
        table1_latex(t1),
    )
    print(t1.to_string())

    # Table 2 — supplementary; collapse points are age-invariant from SYCON data
    print("\nBuilding Table 2 (collapse points — supplementary, age-invariant)...")
    t2 = build_table2(conversations)
    atomic_write_csv(
        os.path.join(tables_dir, "table2_collapse_points.csv"), t2
    )
    atomic_write_text(
        os.path.join(tables_dir, "table2_collapse_points.tex"),
        table2_latex(t2),
    )
    print(t2.to_string())

    # Table 3
    print("\nBuilding Table 3 (dimensional breakdown)...")
    t3 = build_table3(conversations)
    atomic_write_csv(
        os.path.join(tables_dir, "table3_dimension_scores.csv"), t3
    )
    atomic_write_text(
        os.path.join(tables_dir, "table3_dimension_scores.tex"),
        table3_latex(t3),
    )
    print(t3.to_string())

    print(f"\nAll tables written to {tables_dir}")


if __name__ == "__main__":
    run()
