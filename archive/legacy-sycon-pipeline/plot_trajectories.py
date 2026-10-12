"""
Generates publication-quality vulnerability trajectory plots.

One figure per model. Each figure shows three curves (one per age
configuration) of V_t(m, a) across conversation turns, plus a
horizontal line at the collapse threshold theta.

Output
------
results/figures/trajectory_{model}.pdf   (vector, for paper)
results/figures/trajectory_{model}.png   (300 DPI raster, for review)

Usage
-----
    python analysis/plot_trajectories.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "data", "dvf_results.json")
FIGURES_DIR = os.path.join(_ROOT, "results", "figures")
ALPHA_FILE  = os.path.join(_ROOT, "data", "alpha_locked.json")

# ---------------------------------------------------------------------------
# Style constants — do not change without updating paper text
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "gpt4o":         "GPT-4o",
    "claude_sonnet": "Claude Sonnet",
    "gemini_pro":    "Gemini Pro",
}

AGE_LABELS = {
    "a1": "Early (7–9)",
    "a2": "Middle (10–12)",
    "a3": "Late (13–17)",
}

AGE_COLORS = {
    "a1": "#d62728",   # red    — highest vulnerability
    "a2": "#ff7f0e",   # orange — mid vulnerability
    "a3": "#1f77b4",   # blue   — lowest vulnerability
}

AGE_ORDER    = ("a1", "a2", "a3")
LINE_WIDTH   = 1.8
MARKER_SIZE  = 5
MARKER_STYLE = "o"
THETA_COLOR  = "#555555"
THETA_STYLE  = "--"
THETA_WIDTH  = 1.2
FONT_FAMILY  = "serif"
FONT_SIZE    = 11
LABEL_SIZE   = 10
DPI          = 300
FIG_W        = 5.5   # inches — fits single-column ACM layout
FIG_H        = 3.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[dict]:
    """
    Loads dvf_results.json and returns the conversations list.

    Args:
        path: path to dvf_results.json

    Returns:
        list of conversation result dicts
    """
    with open(path) as f:
        data = json.load(f)
    return data["conversations"]


def load_theta(alpha_path: str) -> float:
    """
    Reads the collapse threshold theta from constants.py.

    Args:
        alpha_path: unused (kept for call-site compatibility)

    Returns:
        float: theta
    """
    import sys
    sys.path.insert(0, _ROOT)
    import constants as _c
    return float(_c.THETA)


def aggregate_trajectories(
    conversations: list[dict],
    model: str,
) -> dict[str, dict]:
    """
    Aggregates per-turn trajectories for one model across age configs.

    Where multiple conversations exist for the same (model, age_config),
    trajectories are averaged turn-by-turn. Averaging is performed after
    zero-padding shorter trajectories to the maximum observed length.

    Args:
        conversations: full dvf_results conversations list
        model:         model key (e.g. 'gpt4o')

    Returns:
        dict keyed by age_config ('a1', 'a2', 'a3'), each containing:
            'mean':  np.ndarray of mean V_t per turn
            'se':    np.ndarray of standard error per turn (0 if n=1)
            'n':     number of conversations averaged
            'turns': np.ndarray of 1-indexed turn numbers
    """
    from collections import defaultdict
    buckets: dict[str, list[list[float]]] = defaultdict(list)

    for conv in conversations:
        if conv["model"] != model:
            continue
        buckets[conv["age_config"]].append(conv["trajectory"])

    result = {}
    for age in AGE_ORDER:
        trajs = buckets.get(age, [])
        if not trajs:
            continue
        max_T = max(len(t) for t in trajs)
        padded = np.array([
            t + [t[-1]] * (max_T - len(t))   # pad with last value
            for t in trajs
        ])
        mean = padded.mean(axis=0)
        se   = padded.std(axis=0) / np.sqrt(len(trajs)) if len(trajs) > 1 \
               else np.zeros(max_T)
        result[age] = {
            "mean":  mean,
            "se":    se,
            "n":     len(trajs),
            "turns": np.arange(1, max_T + 1),
        }

    return result


def plot_model(
    model: str,
    trajectories: dict[str, dict],
    theta: float,
    output_dir: str,
) -> None:
    """
    Renders and saves the trajectory figure for one model.

    Args:
        model:        model key
        trajectories: output of aggregate_trajectories()
        theta:        collapse threshold
        output_dir:   directory to save figures
    """
    plt.rcParams.update({
        "font.family":     FONT_FAMILY,
        "font.size":       FONT_SIZE,
        "axes.labelsize":  LABEL_SIZE,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for age in AGE_ORDER:
        if age not in trajectories:
            continue
        data  = trajectories[age]
        turns = data["turns"]
        mean  = data["mean"]
        se    = data["se"]

        ax.plot(
            turns, mean,
            color=AGE_COLORS[age],
            linewidth=LINE_WIDTH,
            marker=MARKER_STYLE,
            markersize=MARKER_SIZE,
            label=AGE_LABELS[age],
        )
        if data["n"] > 1:
            ax.fill_between(
                turns,
                mean - se,
                mean + se,
                color=AGE_COLORS[age],
                alpha=0.15,
            )

    # Collapse threshold
    ax.axhline(
        y=theta,
        color=THETA_COLOR,
        linestyle=THETA_STYLE,
        linewidth=THETA_WIDTH,
        label=f"Collapse threshold (θ={theta:.2f})",
    )

    ax.set_xlabel("Conversation turn")
    ax.set_ylabel(r"$V_t(m, a)$")
    ax.set_title(
        f"Vulnerability trajectory — {MODEL_LABELS.get(model, model)}",
        pad=8,
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=LABEL_SIZE)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, f"trajectory_{model}")
    fig.savefig(stem + ".pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.pdf / .png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    results_path: str = RESULTS_DIR,
    alpha_path:   str = ALPHA_FILE,
    output_dir:   str = FIGURES_DIR,
) -> None:
    """
    Generates trajectory plots for all models in dvf_results.json.

    Args:
        results_path: path to dvf_results.json
        alpha_path:   path to alpha_locked.json (for theta)
        output_dir:   directory where figures are saved
    """
    print("Loading DVF results...")
    conversations = load_results(results_path)

    print("Loading theta...")
    theta = load_theta(alpha_path)

    models = sorted({c["model"] for c in conversations})
    print(f"Models found: {models}")

    for model in models:
        print(f"Plotting {MODEL_LABELS.get(model, model)}...")
        trajectories = aggregate_trajectories(conversations, model)
        if not trajectories:
            print(f"  No trajectories found — skipping.")
            continue
        plot_model(model, trajectories, theta, output_dir)

    print("Done.")


if __name__ == "__main__":
    run()
