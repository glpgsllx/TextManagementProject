"""Figures for Part 1; all analysis stays in statistics.py."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

COLORS = {"custom": "#245D9E", "whitespace": "#B56C22", "wordpunct": "#4F8C60"}


def save_figure(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def make_figures(output, frequencies, comparison, zipf, points, heaps):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "ai6122-part1-meridian", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    ranked = sorted(frequencies["custom"].values(), reverse=True)
    ranks = np.arange(1, len(ranked) + 1)
    fit = zipf["primary_fit"]
    axes[0].loglog(ranks, ranked, color=COLORS["custom"], lw=1.4, label="Observed custom terms")
    left, right = fit["rank_min"], fit["rank_max"]
    xx = np.geomspace(left, right, 200)
    axes[0].loglog(xx, fit["coefficient"] * xx**fit["slope"], "--", color="#C34A36",
                   label=f"OLS: s={fit['exponent_s']:.3f}, R²={fit['r2_log10']:.3f}")
    anchor = ranked[left - 1] * left
    axes[0].loglog(ranks, anchor / ranks, ":", color="#666666", label="Ideal 1/r reference")
    axes[0].set(title="Zipf: full frequency-rank curve", xlabel="Term rank (log scale)", ylabel="Collection frequency (log scale)")
    axes[0].legend(fontsize=8)
    for name in ("custom", "whitespace", "wordpunct"):
        values = sorted(frequencies[name].values(), reverse=True)
        axes[1].loglog(np.arange(1, len(values) + 1), values, color=COLORS[name], label=name, lw=1.2)
    axes[1].set(title="Same corpus, same normalization policy", xlabel="Term rank (log scale)", ylabel="Collection frequency (log scale)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    save_figure(fig, output, "zipf")

    required = [p for p in points if p["required_checkpoint"]]
    f = heaps["four_checkpoint_fit"]
    predictive = heaps["fit_10_30_50_predict_100"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    n = np.array([p["tokens"] for p in points])
    v = np.array([p["types"] for p in points])
    xx = np.geomspace(n.min(), n.max(), 250)
    for ax in axes:
        ax.plot(n, v, ".-", color=COLORS["custom"], label="Nested corpus observations")
        ax.scatter([p["tokens"] for p in required], [p["types"] for p in required], s=45,
                   color=COLORS["custom"], edgecolor="white", zorder=4, label="10%, 30%, 50%, 100% reviews")
        ax.plot(xx, f["k"] * xx**f["beta"], "--", color="#C34A36", label=f"Four-point fit: β={f['beta']:.3f}")
        ax.scatter(n[-1], predictive["predicted_types"], marker="x", s=60, color="#8B579E",
                   label="100% prediction from 10/30/50%")
        ax.grid(alpha=0.15)
        ax.set(xlabel="Cumulative normalized tokens N", ylabel="Vocabulary types V")
    axes[0].set_title("Heaps: vocabulary growth")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
    axes[1].set(xscale="log", yscale="log", title=f"Log-log fit: k={f['k']:.2f}, R²={f['r2_log10']:.4f}")
    axes[0].legend(fontsize=8, loc="lower right")
    save_figure(fig, output, "heaps")

    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    ax.bar([r["method"].replace("_", "\n") for r in comparison], [r["normalized_types"] for r in comparison],
           color=[COLORS[r["method"]] for r in comparison])
    for i, row in enumerate(comparison):
        ax.text(i, row["normalized_types"], f"{row['normalized_types']:,}", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Distinct normalized terms", title="Vocabulary size after common postprocessing")
    ax.margins(y=0.18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
    save_figure(fig, output, "tokenizer_comparison")
