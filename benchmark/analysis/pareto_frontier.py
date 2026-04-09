"""
SAHA-AL Benchmark — Privacy-Utility Pareto Frontier Analysis
==============================================================
Formalizes the Privacy-Utility Score (PUS):
    PUS(λ) = λ · Privacy + (1 - λ) · Utility
where Privacy = (1 - ELR/100) and Utility = BERTScore_F1 / 100.

Produces: Pareto frontier, PUS sweep table, and optional matplotlib figure.

Usage:
  python -m analysis.pareto_frontier \
      --results results/all_eval_results.json \
      --output results/pareto_analysis.json \
      --plot figures/pareto_frontier.png
"""

import argparse
import json
import os

import numpy as np


def compute_pareto_frontier(results_dict: dict):
    """
    Identify Pareto-optimal models on the Privacy-Utility plane.

    Args:
        results_dict: {model_name: {"elr": float, "bertscore": float}}
    Returns:
        (pareto_models, points_array)
    """
    models = list(results_dict.keys())
    points = np.array([
        (1 - r["elr"] / 100, r["bertscore"] / 100)
        for r in results_dict.values()
    ])

    pareto = []
    for i, (px, py) in enumerate(points):
        dominated = any(
            points[j][0] >= px and points[j][1] >= py
            and (points[j][0] > px or points[j][1] > py)
            for j in range(len(points)) if j != i
        )
        if not dominated:
            pareto.append(models[i])

    return pareto, points, models


def sweep_pus(results_dict: dict, lambdas=None):
    """Sweep λ and compute PUS for each model at each operating point."""
    if lambdas is None:
        lambdas = [i / 10 for i in range(11)]

    table = {}
    for name, r in results_dict.items():
        privacy = 1 - r["elr"] / 100
        utility = r["bertscore"] / 100
        table[name] = {
            f"λ={l:.1f}": round(l * privacy + (1 - l) * utility, 4)
            for l in lambdas
        }
    return table


def plot_frontier(results_dict, pareto_models, output_path):
    """Generate a matplotlib scatter plot with Pareto frontier."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        print("[WARN] matplotlib not installed. Skipping plot.")
        return

    SYSTEM_STYLES = {
        "regex":        {"label": "Regex+Faker",     "group": "rule"},
        "spacy":        {"label": "spaCy+Faker",     "group": "rule"},
        "presidio":     {"label": "Presidio",        "group": "rule"},
        "bart-base":    {"label": "BART-base",       "group": "seq2seq"},
        "flan-t5-small":{"label": "Flan-T5-small",   "group": "seq2seq"},
        "t5-small":     {"label": "T5-small",        "group": "seq2seq"},
        "distilbart":   {"label": "DistilBART",      "group": "seq2seq"},
        "t5-eff-tiny":  {"label": "T5-eff-tiny",     "group": "seq2seq"},
    }
    GROUP_COLORS = {"rule": "#E07B39", "seq2seq": "#3B7DD8"}
    GROUP_MARKERS = {"rule": "s", "seq2seq": "o"}

    fig, ax = plt.subplots(1, 1, figsize=(11, 7.5))

    points = {}
    for name, r in results_dict.items():
        x = 1 - r["elr"] / 100
        y = r["bertscore"] / 100
        points[name] = (x, y)

    # Manually resolve label offsets to avoid overlap
    label_offsets = {}
    sorted_names = sorted(points.keys(), key=lambda n: (-points[n][0], -points[n][1]))
    used_positions = []
    for name in sorted_names:
        x, y = points[name]
        ox, oy = 8, 8
        for ux, uy in used_positions:
            if abs(x - ux) < 0.04 and abs(y - uy) < 0.015:
                oy -= 18
        label_offsets[name] = (ox, oy)
        used_positions.append((x, y))

    for name, (x, y) in points.items():
        style = SYSTEM_STYLES.get(name, {"label": name, "group": "seq2seq"})
        group = style["group"]
        is_pareto = name in pareto_models

        color = "#CC2936" if is_pareto else GROUP_COLORS.get(group, "#3B7DD8")
        marker = "*" if is_pareto else GROUP_MARKERS.get(group, "o")
        size = 280 if is_pareto else 120
        edge = "black" if is_pareto else "white"

        ax.scatter(x, y, c=color, s=size, marker=marker, zorder=6,
                   edgecolors=edge, linewidths=0.8)

        ox, oy = label_offsets.get(name, (8, 8))
        ax.annotate(
            style["label"], (x, y),
            textcoords="offset points", xytext=(ox, oy),
            fontsize=9, fontweight="bold" if is_pareto else "normal",
            color=color, alpha=0.95,
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.5)
            if abs(ox) > 8 or abs(oy) > 12 else None,
        )

    pareto_points = sorted(
        [(points[m][0], points[m][1]) for m in pareto_models],
        key=lambda p: p[0],
    )
    if len(pareto_points) > 1:
        px, py = zip(*pareto_points)
        ax.plot(px, py, color="#CC2936", linestyle="--", alpha=0.6,
                linewidth=2, label="Pareto frontier", zorder=4)
    elif len(pareto_points) == 1:
        ax.scatter([], [], c="#CC2936", marker="*", s=200, label="Pareto-optimal")

    for lam in [0.3, 0.5, 0.7]:
        xs = np.linspace(0, 1, 200)
        for target_pus in [0.85, 0.90, 0.95]:
            ys = (target_pus - lam * xs) / (1 - lam)
            valid = (ys >= 0.5) & (ys <= 1.05)
            ax.plot(xs[valid], ys[valid], ":", alpha=0.1, color="gray", linewidth=0.8)

    ax.annotate("← Lower Privacy", xy=(0.15, 0.52), fontsize=8,
                color="gray", alpha=0.6)
    ax.annotate("Higher Privacy →", xy=(0.82, 0.52), fontsize=8,
                color="gray", alpha=0.6)
    ax.annotate("Higher\nUtility ↑", xy=(0.01, 0.96), fontsize=8,
                color="gray", alpha=0.6)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GROUP_COLORS["rule"],
               markersize=10, label="Rule-based"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS["seq2seq"],
               markersize=10, label="Seq2Seq"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#CC2936",
               markersize=14, label="Pareto-optimal"),
        Line2D([0], [0], color="#CC2936", linestyle="--", linewidth=2,
               alpha=0.6, label="Pareto frontier"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10,
              framealpha=0.9, edgecolor="lightgray")

    ax.set_xlabel("Privacy  (1 − ELR)", fontsize=13, labelpad=8)
    ax.set_ylabel("Utility  (BERTScore F1 / 100)", fontsize=13, labelpad=8)
    ax.set_title("Privacy–Utility Pareto Frontier", fontsize=15, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(0.50, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Privacy-Utility Pareto analysis")
    parser.add_argument("--results", required=True,
                        help='JSON with {model: {"elr": float, "bertscore": float}}')
    parser.add_argument("--output", default="results/pareto_analysis.json")
    parser.add_argument("--plot", default=None, help="Path for Pareto plot PNG")
    args = parser.parse_args()

    with open(args.results) as f:
        results_dict = json.load(f)

    pareto_models, points, all_models = compute_pareto_frontier(results_dict)
    pus_table = sweep_pus(results_dict)

    analysis = {
        "pareto_optimal": pareto_models,
        "pus_sweep": pus_table,
        "model_points": {
            m: {"privacy": round(float(p[0]), 4), "utility": round(float(p[1]), 4)}
            for m, p in zip(all_models, points)
        },
    }

    print("\n" + "=" * 50)
    print("  Pareto-Optimal Models:")
    for m in pareto_models:
        r = results_dict[m]
        print(f"    {m}: ELR={r['elr']:.2f}%, BERTScore={r['bertscore']:.2f}")
    print("=" * 50)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to {args.output}")

    if args.plot:
        plot_frontier(results_dict, pareto_models, args.plot)


if __name__ == "__main__":
    main()
