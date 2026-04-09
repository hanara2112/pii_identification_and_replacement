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
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skipping plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for name, r in results_dict.items():
        x = 1 - r["elr"] / 100
        y = r["bertscore"] / 100
        color = "red" if name in pareto_models else "steelblue"
        marker = "*" if name in pareto_models else "o"
        size = 150 if name in pareto_models else 80
        ax.scatter(x, y, c=color, s=size, marker=marker, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.8)

    pareto_points = sorted(
        [(1 - results_dict[m]["elr"] / 100, results_dict[m]["bertscore"] / 100) for m in pareto_models],
        key=lambda p: p[0],
    )
    if len(pareto_points) > 1:
        px, py = zip(*pareto_points)
        ax.plot(px, py, "r--", alpha=0.5, linewidth=1.5, label="Pareto frontier")

    for lam in [0.3, 0.5, 0.7]:
        xs = np.linspace(0, 1, 100)
        for target_pus in [0.7, 0.8, 0.9]:
            ys = (target_pus - lam * xs) / (1 - lam) if lam < 1 else None
            if ys is not None:
                valid = (ys >= 0) & (ys <= 1)
                ax.plot(xs[valid], ys[valid], ":", alpha=0.15, color="gray")

    ax.set_xlabel("Privacy (1 − ELR)", fontsize=12)
    ax.set_ylabel("Utility (BERTScore F1 / 100)", fontsize=12)
    ax.set_title("Privacy-Utility Pareto Frontier", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.5, 1.05)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
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
