"""Figures for Part 3; all analysis stays in sentiment.py."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import SEASONS


COLORS = ("#245D9E", "#B56C22", "#1F9A6B", "#8B579E")
MARKERS = ("o", "s", "^", "D")
INK = "#333333"


def save_figure(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def sector_rows(rows, sector, key):
    return sorted((r for r in rows if r["sector"] == sector), key=key)


def label_line_end(ax, rows, field, sector, offset):
    last = rows[-1]
    ax.annotate(sector, (last["year"], last[field]), xytext=(7, offset), textcoords="offset points",
                color=INK, va="center", fontsize=9)


def set_style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "ai6122-part3-meridian", "savefig.facecolor": "white"})


def make_figures(output, yearly, seasonal, sectors):
    set_style()
    finals = {s: sector_rows(yearly, s, lambda r: r["year"])[-1]["mean_compound"] for s in sectors}
    # Direct labels sit right of the lines; nudge them apart if the lines finish close together.
    gap = {s: (5 if finals[s] == max(finals.values()) else -5) for s in sectors}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for i, sector in enumerate(sectors):
        rows = sector_rows(yearly, sector, lambda r: r["year"])
        years = [r["year"] for r in rows]
        style = {"color": COLORS[i], "marker": MARKERS[i], "markersize": 5, "lw": 2, "label": sector,
                 "markeredgecolor": "white", "markeredgewidth": 0.8}
        axes[0].fill_between(years, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows],
                             color=COLORS[i], alpha=0.14, lw=0)
        axes[0].plot(years, [r["mean_compound"] for r in rows], **style)
        axes[1].plot(years, [r["mean_stars"] for r in rows], **style)
        label_line_end(axes[0], rows, "mean_compound", sector, gap[sector])
        label_line_end(axes[1], rows, "mean_stars", sector, gap[sector])
    axes[0].set(title="VADER sentiment by year (band: 95% bootstrap CI)", xlabel="Year", ylabel="Mean compound score")
    axes[1].set(title="Star rating by year, as an independent reference", xlabel="Year", ylabel="Mean stars", ylim=(1, 5))
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        ax.margins(x=0.12)
    save_figure(fig, output, "sentiment_trend")

    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for i, sector in enumerate(sectors):
        rows = sector_rows(seasonal, sector, lambda r: SEASONS.index(r["season"]))
        means = [r["mean_compound"] for r in rows]
        errors = [[m - r["ci_low"] for m, r in zip(means, rows)], [r["ci_high"] - m for m, r in zip(means, rows)]]
        positions = [j + (i - 0.5) * 0.18 for j in range(len(rows))]
        ax.errorbar(positions, means, yerr=errors, fmt=MARKERS[i], color=COLORS[i], ms=7, lw=1.4, capsize=3,
                    markeredgecolor="white", label=sector)
    ax.set_xticks(range(len(SEASONS)), SEASONS)
    ax.set(title="VADER sentiment by season, all selected years pooled (bars: 95% CI)", ylabel="Mean compound score")
    ax.grid(alpha=0.15, axis="y")
    ax.legend(fontsize=8, frameon=False)
    save_figure(fig, output, "sentiment_seasonal")


def make_drift_figures(output, drift, themes, terms):
    set_style()
    fig, axes = plt.subplots(1, len(terms), figsize=(11, 4.2), layout="constrained", sharey=True)
    for ax, term in zip(axes, terms):
        rows = [r for r in drift if r["term"] == term]
        later = rows[1:]
        ax.plot([r["period"] for r in rows], [r["same_period_resample_jaccard"] for r in rows], color=INK, ls="--", lw=1.4,
                label="Same year, bootstrap copy (noise reference)")
        ax.plot([r["period"] for r in later], [r["jaccard_previous"] for r in later], color=COLORS[0], marker="o", lw=2,
                markeredgecolor="white", label="Against previous year")
        ax.plot([r["period"] for r in later], [r["jaccard_first"] for r in later], color=COLORS[1], marker="s", lw=2,
                markeredgecolor="white", label=f"Against {rows[0]['period']}")
        ax.set(title=f'Top-20 overlap for "{term}"', xlabel="Year", ylim=(0, 1))
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    axes[0].set_ylabel("Jaccard overlap")
    save_figure(fig, output, "pmi_drift")

    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    names = list(dict.fromkeys(r["theme"] for r in themes))
    for i, theme in enumerate(names):
        rows = [r for r in themes if r["theme"] == theme]
        years = [r["year"] for r in rows]
        ax.fill_between(years, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], color=COLORS[i], alpha=0.12, lw=0)
        ax.plot(years, [r["per_100_mentions"] for r in rows], color=COLORS[i], marker=MARKERS[i], markersize=5, lw=2,
                markeredgecolor="white", markeredgewidth=0.8, label=theme)
    ax.set(title=f'Theme words near "{themes[0]["term"]}" (band: 95% bootstrap CI)', xlabel="Year",
           ylabel="Co-occurrences per 100 mentions", ylim=(0, None))
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, frameon=False, ncols=len(names), loc="center", bbox_to_anchor=(0.5, 0.6))
    save_figure(fig, output, "pmi_themes")
