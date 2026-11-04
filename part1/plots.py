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


def make_heaps_figure(output, points, heaps):
    """Single-column figure: 3.35 inches wide, with 9-point text at that size."""
    style = {"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9,
             "ytick.labelsize": 9, "legend.fontsize": 9, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fit = heaps["fit"]
        prediction = heaps["prediction"]
        n = np.array([p["tokens"] for p in points])
        v = np.array([p["types"] for p in points])
        review_points = [p for p in points if p["review_fraction"] != ""]
        xx = np.geomspace(fit["N_min"], fit["N_max"], 300)
        fig, ax = plt.subplots(figsize=(3.35, 2.85), layout="constrained")
        ax.loglog(n, v, color=COLORS["custom"], lw=1.3, label="Observed")
        ax.loglog(xx, fit["k"] * xx**fit["beta"], "--", color="#C34A36",
                  lw=1.2, label="Power-law fit")
        ax.scatter([p["tokens"] for p in review_points], [p["types"] for p in review_points],
                   s=15, color=COLORS["custom"], zorder=3, label="10/30/50/100% reviews")
        ax.scatter(n[-1], prediction["predicted_types"], marker="x", s=28,
                   color="#825AA8", zorder=4, label="Full-size prediction")
        ax.text(0.97, 0.05,
                f"k = {fit['k']:.3f}\nβ = {fit['beta']:.4f}\nR² = {fit['r2_log10']:.4f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
        ax.set(xlabel="Tokens, N", ylabel="Vocabulary, V")
        ax.grid(alpha=0.15)
        ax.legend(loc="upper left", frameon=False, handlelength=1.6,
                  handletextpad=0.5, borderaxespad=0.3, labelspacing=0.25)
        # Keep the physical width fixed, so LaTeX column-width placement retains 9 pt text.
        fig.savefig(output / "heaps.png", dpi=300, bbox_inches=None)
        fig.savefig(output / "heaps.svg", bbox_inches=None, metadata={"Date": None})
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

    make_heaps_figure(output, points, heaps)

    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    ax.bar([r["method"].replace("_", "\n") for r in comparison], [r["normalized_types"] for r in comparison],
           color=[COLORS[r["method"]] for r in comparison])
    for i, row in enumerate(comparison):
        ax.text(i, row["normalized_types"], f"{row['normalized_types']:,}", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Distinct normalized terms", title="Vocabulary size after common postprocessing")
    ax.margins(y=0.18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
    save_figure(fig, output, "tokenizer_comparison")
