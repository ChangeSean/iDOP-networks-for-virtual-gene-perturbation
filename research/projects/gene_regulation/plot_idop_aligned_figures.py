"""Draw the genome-wide workflow and matching leave-one-target-out analysis."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from plot_idop_full_decomposition import CASES, BLUE, PURPLE, RUST, decomposition
from plot_prdm14_dnmt3b_manuscript import OUT, INK, CONTROL, PALE, arrow, blank, box
from screen_idop_transcriptome_readouts import DAYS, ROOT, REAL, OUT as DATA, load_states, fit_response


def save(fig, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=350)
    plt.close(fig)


def workflow(protocol, readouts, full):
    screen = pd.read_csv(DATA / "external_screen.csv")
    sources = len(protocol["external_guides"])
    assert len(full) == sources * len(readouts)
    eligible = full.eligible.sum()
    assert full.loc[full.nominated, "eligible"].all()
    assert len(screen) == full.nominated.sum()
    passes = screen.external_status.eq("pass").sum()
    assert (len(full), len(screen), passes) == (62300, 20, 7)
    fig, ax = plt.subplots(figsize=(180 / 25.4, 58 / 25.4))
    fig.subplots_adjust(left=.026, right=.974, bottom=.052, top=.97)
    blank(ax)
    stages = [
        ("a", "Fit the weighted network", "GSE213069 · days 2–5",
         f"{len(protocol['targets'])} targets · two guides each\n{len(protocol['core'])} core genes · {len(readouts):,} readouts"),
        ("b", "Simulate perturbations", f"{len(full):,} source–readout pairs",
         f"{sources} sources × {len(readouts):,} readouts\nRank responses after 3 days of simulation"),
        ("c", "Freeze nominations", f"{len(screen)} nominated pairs",
         f"{eligible:,} met magnitude and sign filters\n"
         f"Top {protocol['maximum_nominees_per_source']} by magnitude per source"),
        ("d", "Check independent responses", f"{passes} passed the first screen",
         "GSE303158 · two guides, two batches\nFollow-up across human assays"),
    ]
    for i, (letter, title, primary, detail) in enumerate(stages):
        x, y = .015 + .505 * (i % 2), .57 if i < 2 else .03
        ax.text(x, y + .35, letter, fontsize=8, weight="bold")
        ax.text(x + .042, y + .35, title, fontsize=7, weight="bold")
        box(ax, x, y, .465, .30, primary, detail, face=PALE if i >= 2 else "#f2f5f6", title_size=7)
    arrow(ax, (.485, .705), (.515, .705))
    arrow(ax, (.485, .16), (.515, .16))
    ax.plot([.75, .75, .25], [.56, .47, .47], color=CONTROL, lw=1.1)
    arrow(ax, (.25, .47), (.25, .405))
    save(fig, "Figure_1_genomewide_nomination")


def heldout(protocol, full):
    core, targets = protocol["core"], protocol["targets"]
    source_ids = np.tile([core.index(s) for s in targets], 2)
    labels = full.set_index(["source", "marker"])
    with TemporaryDirectory(prefix="genomewide_fig3_", dir=ROOT / "runtime_temp") as temp:
        with threadpool_limits(1):
            states, readouts, _ = load_states(core, targets, Path(temp) / "readouts.csv")
            scores_by_fold = []
            for source in dict.fromkeys(s for s, _ in CASES):
                j = core.index(source)
                train = source_ids != j
                assert train.sum() == 30 and len(readouts) == 15575
                scores, _, _, matrix = fit_response(states[train], core, source_ids[train], [j])
                scores = scores[0]
                ranks = np.argsort(np.argsort(-np.abs(scores), kind="stable")) + 1
                scores_by_fold.append(pd.DataFrame(dict(source=source, readout=readouts,
                                                         heldout_rank=ranks, heldout_response=scores)))
                if source == "PRDM14":
                    dnmt = scores[np.searchsorted(readouts, "DNMT3B")]
                    trajectory = decomposition(np.r_[core, readouts], matrix, source, "DNMT3B", dnmt)
    rank_table = pd.concat(scores_by_fold, ignore_index=True)
    indexed = rank_table.set_index(["source", "readout"])
    cases = indexed.loc[list(CASES)].reset_index()
    cases["full_rank"] = [labels.loc[pair, "rank"] for pair in CASES]
    cases["full_response"] = [labels.loc[pair, "day5_response"] for pair in CASES]
    assert (cases.heldout_response < 0).all()
    assert cases.heldout_rank.tolist() == [18, 1, 3, 4, 3791]
    observed = collect_observed_holdout(states, np.r_[core, readouts], targets)
    return rank_table, cases, trajectory, observed


def collect_observed_holdout(states, genes, targets):
    gene_index = pd.Index(genes)
    records = []
    for source, readout in CASES:
        source_index = targets.index(source)
        readout_index = gene_index.get_loc(readout)
        for arm in (1, 2):
            guide = f"{source}_{arm}"
            trajectory = states[source_index + (arm - 1) * len(targets), :, readout_index]
            records.extend(dict(source=source, readout=readout, guide=guide,
                                day=int(day), observed_contrast=value)
                           for day, value in zip(DAYS, trajectory, strict=True))
    return pd.DataFrame(records)


def draw_observed_holdout(axes, observed):
    guide_styles = ((BLUE, "o", "-"), (PURPLE, "s", "--"))
    for ax, (source, readout) in zip(axes, CASES, strict=True):
        ax.axhline(0, color="#AAB5C6", lw=.7, ls=":", zorder=0)
        for arm, (color, marker, linestyle) in enumerate(guide_styles, start=1):
            rows = observed.loc[observed.guide.eq(f"{source}_{arm}") & observed.readout.eq(readout)]
            rows = rows.sort_values("day")
            np.testing.assert_array_equal(rows.day, DAYS)
            ax.plot(rows.day, rows.observed_contrast, color=color, marker=marker, ls=linestyle,
                    lw=1.15, ms=3, markerfacecolor="white", markeredgewidth=.85)
        ax.set(xticks=DAYS, xlim=(1.76, 5.24), ylim=(-1.48, .11), yticks=[-1.4, -.7, 0])
        ax.set_title(readout, fontsize=7.5)
        ax.tick_params(length=2.4)
    axes[0].set_ylabel("Observed contrast")
    return [Line2D([], [], color=color, marker=marker, ls=linestyle, lw=1.1, ms=3,
                   markerfacecolor="white", label=f"Guide {i}")
            for i, (color, marker, linestyle) in enumerate(guide_styles, start=1)]


def plot_heldout(cases, trajectory, full_trajectory, observed):
    fig = plt.figure(figsize=(180 / 25.4, 192 / 25.4))
    ax = fig.add_axes([.144, .731, .811, .200])
    y = np.arange(len(cases))
    ax.axvline(0, color="#AAB5C6", lw=.6, zorder=0)
    ax.hlines(y, np.minimum(cases.full_response, cases.heldout_response),
              np.maximum(cases.full_response, cases.heldout_response), color="#B3BED0", lw=1.05)
    ax.scatter(cases.full_response, y, s=26, facecolor="white", edgecolor=BLUE, zorder=3, label="Full fit")
    ax.scatter(cases.heldout_response, y, s=24, color=PURPLE, zorder=4, label="Target guides omitted")
    ax.set(xlim=(-.325, .025), xticks=[-.3, -.2, -.1, 0], yticks=y, yticklabels=cases.readout,
           ylim=(4.55, -.55), xlabel="Simulated expression contrast at 3 d")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(.50, -.28), ncol=2, fontsize=6.5,
              handletextpad=.25, columnspacing=.8)

    ax = fig.add_axes([.119, .337, .321, .218])
    elapsed = trajectory.day - trajectory.day.iloc[0]
    ax.axhline(0, color="#AAB5C6", lw=.6, zorder=0)
    for gene in trajectory.columns[3:].drop("PRDM14"):
        ax.plot(elapsed, trajectory[gene], color="#A7B4C5", lw=.55, alpha=.75)
    for column, label, color, style in (("self_DNMT3B", "Self", RUST, "-"),
                                        ("total", "Total", BLUE, "-"),
                                        ("PRDM14", "PRDM14", PURPLE, "--")):
        ax.plot(elapsed, trajectory[column], color=color, lw=1.25, ls=style)
        ax.annotate(label, (elapsed.iloc[-1], trajectory[column].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=color, fontsize=6.4,
                    annotation_clip=False)
    ax.set(xlim=(-.04, 3.12), xticks=[0, 1, 2, 3], xlabel="Elapsed simulation time (days)")
    ax.set_title("DNMT3B", loc="left", fontsize=7, weight="bold", pad=6)

    endpoint_rows = []
    for fit_name, terms in (("Full", full_trajectory), ("Held out", trajectory)):
        end = terms.iloc[-1]
        parts = {"PRDM14": end.PRDM14, "Self": end.self_DNMT3B,
                 "Other inputs": end.iloc[3:].drop("PRDM14").sum(), "Total": end.total}
        np.testing.assert_allclose(sum(list(parts.values())[:3]), parts["Total"], atol=2e-12)
        endpoint_rows.extend(dict(fit=fit_name, term=term, contribution=value) for term, value in parts.items())
    endpoints = pd.DataFrame(endpoint_rows)
    contrast = endpoints.pivot(index="term", columns="fit", values="contribution").loc[
        ["PRDM14", "Self", "Other inputs", "Total"]]
    ax = fig.add_axes([.640, .337, .315, .218])
    y = np.arange(len(contrast))
    ax.axvline(0, color="#AAB5C6", lw=.6, zorder=0)
    ax.hlines(y, contrast.min(axis=1), contrast.max(axis=1), color="#B3BED0", lw=1.)
    ax.scatter(contrast["Full"], y, s=27, facecolor="white", edgecolor=BLUE, zorder=3)
    ax.scatter(contrast["Held out"], y, s=19, color=PURPLE, zorder=4)
    ax.set(xlim=(-.71, .43), xticks=[-.6, -.4, -.2, 0, .2, .4],
           yticks=y, yticklabels=contrast.index, ylim=(3.55, -.55), xlabel="Cumulative contribution at 3 d")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    grid = fig.add_gridspec(1, len(CASES), left=.119, right=.955, bottom=.088, top=.207, wspace=.23)
    observed_axes = [fig.add_subplot(grid[0, 0])]
    observed_axes.extend(fig.add_subplot(grid[0, i], sharey=observed_axes[0]) for i in range(1, len(CASES)))
    handles = draw_observed_holdout(observed_axes, observed)
    for observed_ax in observed_axes[1:]:
        observed_ax.tick_params(labelleft=False)
    centers = [(p.x0 + p.x1) / 2 for p in (a.get_position() for a in observed_axes)]
    fig.text(np.mean(centers[:4]), .253, "PRDM14 guides", ha="center", fontsize=7, weight="bold")
    fig.text(centers[-1], .253, "POU5F1 guides", ha="center", fontsize=7, weight="bold")
    fig.text(.537, .043, "Experimental day", ha="center", fontsize=7)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.537, .005), frameon=False,
               ncol=2, handletextpad=.36, columnspacing=1.0, fontsize=6.5)

    for ax in fig.axes:
        ax.tick_params(labelsize=6.5, width=.55)
        ax.xaxis.label.set_size(7)
    for label, x, y in (("a", .036, .954), ("b", .036, .590), ("c", .510, .590), ("d", .036, .253)):
        fig.text(x, y, label, fontsize=8, weight="bold", color=INK)
    save(fig, "Figure_3_genomewide_heldout")
    endpoints.to_csv(OUT / "Figure_3_DNMT3B_day5_comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replot", action="store_true", help="Reuse the saved held-out fits for Fig. 3.")
    args = parser.parse_args()
    protocol = json.loads((DATA / "protocol.json").read_text(encoding="utf-8"))
    full = pd.read_csv(DATA / "all_scored_pairs.csv")
    readouts = pd.read_csv(DATA / "eligible_readouts.csv")
    cohort = json.loads((REAL / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["targets"] == cohort["targets"] and protocol["core"] == cohort["all_targets"]
    assert protocol["forcing"] == -1 and protocol["stability_margin"] == .01
    observed_path = OUT / "Figure_3_observed_holdout_trajectories.csv"
    if args.replot:
        scores = pd.read_csv(OUT / "Figure_3_genomewide_heldout_scores.csv")
        cases = pd.read_csv(OUT / "Figure_3_case_ranks.csv")
        trajectory = pd.read_csv(OUT / "Figure_3_DNMT3B_heldout_terms.csv")
        if observed_path.is_file():
            observed = pd.read_csv(observed_path)
        else:
            core, targets = protocol["core"], protocol["targets"]
            with TemporaryDirectory(prefix="genomewide_fig3_observed_", dir=ROOT / "runtime_temp") as temp:
                states, genes, _ = load_states(core, targets, Path(temp) / "readouts.csv")
            observed = collect_observed_holdout(states, np.r_[core, genes], targets)
    else:
        workflow(protocol, readouts, full)
        scores, cases, trajectory, observed = heldout(protocol, full)
        scores.to_csv(OUT / "Figure_3_genomewide_heldout_scores.csv", index=False)
        cases.to_csv(OUT / "Figure_3_case_ranks.csv", index=False)
        trajectory.to_csv(OUT / "Figure_3_DNMT3B_heldout_terms.csv", index=False)
    full_trajectory = pd.read_csv(OUT / "Figure_S4_DNMT3B_decomposition_curves.csv")
    matched = scores.merge(full.rename(columns={"marker": "readout"}), on=["source", "readout"],
                           validate="one_to_one")
    assert len(matched) == len(scores) == 2 * len(readouts)
    agreement = []
    for source, group in matched.groupby("source", sort=False):
        assert len(group) == len(readouts)
        rho = float(spearmanr(group.day5_response, group.heldout_response).statistic)
        agreement.append(dict(source=source, readouts=len(group), spearman_rho=rho))
    pd.DataFrame(agreement).to_csv(OUT / "Figure_3_rank_concordance.csv", index=False)
    assert len(observed) == 40 and np.isfinite(observed.observed_contrast).all()
    observed.to_csv(observed_path, index=False)
    plot_heldout(cases, trajectory, full_trajectory, observed)


if __name__ == "__main__":
    main()
