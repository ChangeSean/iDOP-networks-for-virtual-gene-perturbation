"""Pair virtual responses with omitted observations and independent assays."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from build_followup_figures_20261112 import (
    BLUE, DEPLETION, FOLLOWUP, GREY, LIGHT, SHRNA, export, replicate_groups,
)
from plot_idop_full_decomposition import PURPLE
from plot_prdm14_dnmt3b_manuscript import OUT


def label(ax, letter, title):
    ax.set_title(title, fontsize=8, pad=7)
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-15, 7),
                textcoords="offset points", fontsize=9, weight="bold")


def zero(ax, axis="y"):
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(0, color=LIGHT, ls=":", lw=.75, zorder=0)


def comparison(ax, full, held, y):
    zero(ax, "x")
    ax.hlines(y, full, held, color="#BCC8D9", lw=1.1)
    ax.scatter(full, y, facecolors="white", edgecolors=BLUE, s=25, zorder=3)
    ax.scatter(held, y, color=PURPLE, s=23, zorder=4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def fit_key(ax):
    full = Line2D([], [], marker="o", lw=0, markerfacecolor="white", markeredgecolor=BLUE, ms=4)
    held = Line2D([], [], marker="o", lw=0, color=PURPLE, ms=4)
    ax.legend([full, held], ["Full fit", "Target omitted"], ncol=2, fontsize=6.7,
              loc="upper center", bbox_to_anchor=(.48, -.31), handletextpad=.2, columnspacing=.7)


def guide_panel(ax, observed, readout, show_key=False):
    zero(ax)
    groups = observed.loc[observed.readout.eq(readout)].groupby("guide")
    for i, ((_, rows), color, marker, style) in enumerate(
        zip(groups, (BLUE, PURPLE), ("o", "s"), ("-", "--"), strict=True), 1
    ):
        rows = rows.sort_values("day")
        np.testing.assert_array_equal(rows.day, [2, 3, 4, 5])
        ax.plot(rows.day, rows.observed_contrast, color=color, marker=marker, ls=style,
                lw=1.1, ms=3.5, markerfacecolor="white", label=f"Guide {i}")
    ax.set(xlim=(1.78, 5.22), ylim=(-1.48, .14), xticks=[2, 3, 4, 5],
           yticks=[-1.4, -.7, 0], xlabel="Experimental day", ylabel="Observed contrast")
    if show_key:
        ax.legend(loc="lower left", ncol=2, fontsize=6.3, handlelength=1.2,
                  handletextpad=.3, columnspacing=.6, bbox_to_anchor=(-.03, -.01))


def dnmt3b(observed):
    terms = pd.read_csv(OUT / "Figure_3_DNMT3B_day5_comparison.csv")
    terms = terms.pivot(index="term", columns="fit", values="contribution").loc[
        ["PRDM14", "Self", "Other inputs", "Total"]
    ]
    np.testing.assert_allclose(terms.iloc[:3].sum(), terms.iloc[3])
    crispr = pd.read_csv(OUT / "Table_S2A_primed_CRISPR.csv")
    probes = pd.read_csv(OUT / "Table_S2B_shRNA_probes.csv")
    regions = pd.read_csv(OUT / "Table_S2C_naive_RNA_regions.csv")
    assert (len(crispr), len(probes), len(regions)) == (4, 5, 29)

    fig = plt.figure(figsize=(180 / 25.4, 4.38))
    fig.text(.51, .969, "PRDM14 → DNMT3B", ha="center", fontsize=9, weight="bold")
    grid = fig.add_gridspec(2, 6, left=.105, right=.970, top=.864, bottom=.121,
                            hspace=.89, wspace=1.05, height_ratios=[1.12, 1])
    ax = fig.add_subplot(grid[0, :2])
    label(ax, "a", "Virtual PRDM14 suppression")
    comparison(ax, terms.Full, terms["Held out"], [3, 2, 1, 0])
    ax.set(yticks=[3, 2, 1, 0], yticklabels=terms.index, xlim=(-.71, .43),
           xticks=[-.6, -.3, 0, .3], ylim=(-.5, 3.55), xlabel="Contribution at 3 d")
    fit_key(ax)

    ax = fig.add_subplot(grid[0, 2:4])
    label(ax, "b", "Omitted PRDM14 guides")
    guide_panel(ax, observed, "DNMT3B", show_key=True)

    ax = fig.add_subplot(grid[0, 4:])
    label(ax, "c", "External CRISPR")
    zero(ax, "x")
    for (batch, rows), marker, color, offset in zip(
        crispr.groupby("batch"), ("o", "s"), (BLUE, "#5987BB"), (.12, -.12), strict=True
    ):
        y = rows.guide.str.endswith("7").astype(int) + offset
        ax.scatter(rows.delta_log1p_cp10k, y, marker=marker, color=color, s=22, label=f"Batch {batch}")
    ax.set(yticks=[1, 0], yticklabels=["sg_7", "sg_8"], ylim=(-.38, 1.39),
           xlim=(-.59, .04), xticks=[-.5, -.25, 0], xlabel="Contrast (log1p CP10k)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.31), ncol=2, fontsize=6.7,
              handletextpad=.15, columnspacing=.6)

    top = fig.add_subplot(grid[1, :4])
    top.set_axis_off()
    label(top, "d", "External shRNA")
    probe_grid = grid[1, :4].subgridspec(1, 5, wspace=.74)
    for i, row in enumerate(probes.filter(regex=r"^GSM").to_numpy()):
        ax = fig.add_subplot(probe_grid[0, i])
        zero(ax)
        for j, (group, color, marker) in enumerate(zip(
            (row[:3], row[3:]), (GREY, SHRNA), ("o", "D"), strict=True
        )):
            centered = group - row[:3].mean()
            ax.scatter(j + np.array([-.09, 0, .09]), centered, color=color, marker=marker,
                       edgecolors="white", linewidths=.4, s=17, zorder=3)
            ax.hlines(centered.mean(), j - .21, j + .21, color=color, lw=1.1)
        ax.set(xlim=(-.38, 1.38), xticks=[], xlabel=f"Probe {i+1}")
        ax.yaxis.set_major_locator(MaxNLocator(3))
        if i == 0:
            ax.set_ylabel("Centered intensity")

    ax = fig.add_subplot(grid[1, 4:])
    label(ax, "e", "External protein depletion")
    zero(ax)
    ax.scatter(range(1, 30), np.sort(regions.IAA_minus_control_mean_log2RPM), color=DEPLETION, s=12, lw=0)
    ax.set(xlabel="Ranked regions", ylabel="Contrast (log2 RPM)", xticks=[1, 15, 29],
           xlim=(-.4, 30.4), ylim=(-1.42, .12))
    fig.legend(handles=[
        Line2D([], [], marker="o", lw=0, color=GREY, ms=4, label="Control"),
        Line2D([], [], marker="D", lw=0, color=SHRNA, ms=4, label="Knockdown"),
    ], loc="lower center", bbox_to_anchor=(.38, .006), ncol=2, fontsize=7,
        handletextpad=.25, columnspacing=.8)
    for ax in fig.axes:
        ax.tick_params(labelsize=7, length=2.3)
        ax.xaxis.label.set_size(7.4)
        ax.yaxis.label.set_size(7.4)
    return fig


def followup_cases(observed, cases):
    features = pd.read_csv(FOLLOWUP / "feature_values.csv")
    effects = pd.read_csv(FOLLOWUP / "feng_published_effects.csv")
    fig = plt.figure(figsize=(180 / 25.4, 5.05))
    grid = fig.add_gridspec(2, 3, left=.103, right=.970, top=.880, bottom=.121,
                            wspace=.60, hspace=.68)
    for i, (source, readout) in enumerate((("PRDM14", "TDGF1"), ("POU5F1", "RARRES2"))):
        row = cases.loc[(source, readout)]
        assert row.full_response < 0 and row.heldout_response < 0
        row_top = grid[i, 0].get_position(fig).y1
        fig.text(.51, row_top + .085, f"{source} → {readout}", ha="center", fontsize=9, weight="bold")
        local = grid[i, 0].subgridspec(2, 1, height_ratios=[.50, 1], hspace=.88)
        ax = fig.add_subplot(local[0])
        label(ax, "ad"[i], "Model and holdout")
        comparison(ax, row.full_response, row.heldout_response, 0)
        ax.set(xlim=(-.325, .026), xticks=[-.3, -.15, 0], yticks=[], ylim=(-.3, .3),
               xlabel="Simulated contrast at 3 d")
        ax = fig.add_subplot(local[1])
        guide_panel(ax, observed, readout)

        data = features.query("source == @source and gene == @readout")
        ax = fig.add_subplot(grid[i, 1])
        label(ax, "be"[i], "External shRNA")
        if i == 0:
            ax.set_axis_off()
            probes = data.query("dataset == 'GSE22795'")
            assert len(probes) == 2
            sub = grid[i, 1].subgridspec(1, 2, wspace=.63)
            for j, probe in enumerate(probes.itertuples(index=False)):
                ax = fig.add_subplot(sub[0, j])
                zero(ax)
                replicate_groups(ax, 0, probe)
                ax.set(xlim=(-.35, .35), xticks=[0], xticklabels=[f"Probe {j+1}"])
                ax.yaxis.set_major_locator(MaxNLocator(3))
                if j == 0:
                    ax.set_ylabel("Centered intensity")
        else:
            shrna = data.query("dataset == 'GSE21135'").sort_values("time")
            assert len(shrna) == 2
            zero(ax)
            for j, sample in enumerate(shrna.itertuples(index=False)):
                replicate_groups(ax, j, sample)
            ax.set(xlim=(-.36, 1.36), xticks=[0, 1], xticklabels=["Day 2", "Day 3"],
                   ylabel="Centered intensity")

        ax = fig.add_subplot(grid[i, 2])
        if i == 0:
            label(ax, "c", "External protein depletion")
            regions = data.query("dataset == 'GSE164786'")
            assert len(regions) == 14
            zero(ax)
            ax.scatter(range(1, 15), np.sort(regions.delta), color=DEPLETION, s=15, lw=0)
            ax.set(xlabel="Ranked regions", ylabel="Contrast (log2 RPM)", xticks=[1, 7, 14],
                   xlim=(.15, 14.85), ylim=(-1.18, .09))
        else:
            label(ax, "f", "External CRISPRi")
            rows = effects.query("Target == @source and Expressed_Gene_Symbol == @readout").set_index("screen")
            contrasts = rows.loc[["GenomeWideScreen", "TargetedScreen"], "lfc"]
            zero(ax, "x")
            ax.hlines([1, 0], contrasts, 0, color="#C3D2EA", lw=1.15)
            ax.scatter(contrasts, [1, 0], color=BLUE, s=23, lw=0)
            ax.set(yticks=[1, 0], yticklabels=["Genome-wide", "Targeted"], ylim=(-.4, 1.4),
                   xlim=(-.41, .045), xticks=[-.4, -.2, 0], xlabel="Published log fold change")
            ax.tick_params(axis="y", length=0, pad=2)
            ax.spines["left"].set_visible(False)
    for ax in fig.axes:
        ax.tick_params(labelsize=7, length=2.3)
        ax.xaxis.label.set_size(7.4)
        ax.yaxis.label.set_size(7.4)
    symbols = [("Full fit", BLUE, "o", "white", ""), ("Target omitted", PURPLE, "o", PURPLE, ""),
               ("Guide 1", BLUE, "o", "white", "-"), ("Guide 2", PURPLE, "s", "white", "--"),
               ("Control", GREY, "o", GREY, ""), ("Knockdown", SHRNA, "D", SHRNA, "")]
    handles = [Line2D([], [], label=name, color=color, marker=marker, markerfacecolor=face, ls=style, ms=4)
               for name, color, marker, face, style in symbols]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .006), ncol=6,
               handletextpad=.35, columnspacing=.9, handlelength=1.45, fontsize=7)
    return fig


def main():
    observed = pd.read_csv(OUT / "Figure_3_observed_holdout_trajectories.csv")
    cases = pd.read_csv(OUT / "Figure_3_case_ranks.csv").set_index(["source", "readout"])
    export(dnmt3b(observed), OUT, "Figure_3_DNMT3B_paired_evidence")
    export(followup_cases(observed, cases), OUT, "Figure_4_TDGF1_RARRES2_paired_evidence")


if __name__ == "__main__":
    main()
