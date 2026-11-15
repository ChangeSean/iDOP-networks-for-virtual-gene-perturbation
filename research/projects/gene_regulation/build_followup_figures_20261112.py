"""Plot independent candidate responses and screen signed Hallmark gene programs."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/idop_feasibility_20261109"
FOLLOWUP = RUN / "independent_response_followup_20261110"
FENG = ROOT / "data/Feng_iPSC_CRISPRi_figshare/effects/TargetedScreen_LFC_byGene.tsv.gz"
GMT = ROOT / "data/gene_sets/MSigDB_Hallmark_2020_Enrichr.gmt"
OUT = ROOT / "outputs/idop_followup_figures_20261112"
STAGE = ROOT / "runtime_temp/followup_figure_pilot_20261112"
SOURCES = ("PRDM14", "POU5F1")
SCREEN_Q = .10
BLUE, GREY, INK, LIGHT = "#3156A3", "#7A7F89", "#202631", "#ABB8C9"
SHRNA, DEPLETION = "#B45337", "#76518D"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": LIGHT, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})


def export(fig, directory, name):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=350)
    plt.close(fig)


def replicate_groups(ax, center, row, distance=.18):
    ctrl = np.asarray(list(json.loads(row.control_values).values()))
    treated = np.asarray(list(json.loads(row.treated_values).values()))
    np.testing.assert_allclose(treated.mean() - ctrl.mean(), row.delta, atol=1e-10)
    reference = ctrl.mean()
    for x, values, color, marker in ((center - distance, ctrl, GREY, "o"),
                                     (center + distance, treated, SHRNA, "D")):
        centered = values - reference
        xj = x + np.linspace(-.045, .045, len(centered))
        ax.scatter(xj, centered, color=color, marker=marker, s=19, edgecolors="white",
                   linewidths=.45, zorder=3)
        ax.hlines(centered.mean(), x - .108, x + .108, color=color, lw=1.15, zorder=2)


def figure5():
    values = pd.read_csv(FOLLOWUP / "feature_values.csv")
    probes = values.query("dataset == 'GSE22795' and source == 'PRDM14' and gene == 'TDGF1'")
    regions = values.query("dataset == 'GSE164786' and source == 'PRDM14' and gene == 'TDGF1'")
    shrna = values.query("dataset == 'GSE21135' and source == 'POU5F1' and gene == 'RARRES2'").sort_values("time")
    effects = pd.read_csv(FOLLOWUP / "feng_published_effects.csv")
    crispri = effects.query("Target == 'POU5F1' and Expressed_Gene_Symbol == 'RARRES2'").set_index("screen")
    assert (len(probes), len(regions), len(shrna), len(crispri)) == (2, 14, 2, 2)
    assert pd.concat([probes, regions, shrna]).delta.lt(0).all() and crispri.lfc.lt(0).all()
    assert probes.complete_expected_separation.sum() == 1
    assert regions.complete_expected_separation.sum() == 13 and shrna.complete_expected_separation.all()

    fig = plt.figure(figsize=(180 / 25.4, 4.55))
    grid = fig.add_gridspec(2, 2, left=.128, right=.959, top=.831, bottom=.137, wspace=.42, hspace=.81)
    for y, label in ((.969, "TDGF1 after PRDM14 perturbation"), (.473, "RARRES2 after POU5F1 perturbation")):
        fig.text(.50, y, label, ha="center", va="center", fontsize=9, fontweight="bold")
    for x, y, letter in ((.023, .892, "a"), (.512, .892, "b"), (.023, .416, "c"), (.512, .416, "d")):
        fig.text(x, y, letter, va="center", fontsize=9, fontweight="bold")

    small = grid[0, 0].subgridspec(1, 2, wspace=.57)
    for i, row in enumerate(probes.itertuples(index=False)):
        ax = fig.add_subplot(small[0, i])
        ax.axhline(0, color=LIGHT, lw=.75, ls="--", zorder=0)
        replicate_groups(ax, 0., row)
        ax.set(xlim=(-.4, .4), xticks=[0], xticklabels=[f"Probe {i + 1}"])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="x", length=0, pad=4)
        if i == 0:
            ax.set_ylabel("Centered intensity")
    fig.text(.128, .892, "shRNA knockdown", va="center")

    ax = fig.add_subplot(grid[0, 1])
    ax.axhline(0, color=LIGHT, lw=.75, ls="--", zorder=0)
    ax.scatter(np.arange(1, len(regions) + 1), np.sort(regions.delta), color=DEPLETION, s=15, lw=0)
    ax.set(xticks=[1, 7, 14], xlim=(.15, 14.85), ylim=(-1.18, .09),
           xlabel="Ranked regions", ylabel="Contrast (log2 RPM)")
    fig.text(.614, .892, "Protein depletion", va="center")

    ax = fig.add_subplot(grid[1, 0])
    ax.axhline(0, color=LIGHT, lw=.75, ls="--", zorder=0)
    for i, row in enumerate(shrna.itertuples(index=False)):
        replicate_groups(ax, i, row)
    ax.set(xticks=[0, 1], xticklabels=["Day 2", "Day 3"], xlim=(-.39, 1.39), ylabel="Centered intensity")
    ax.tick_params(axis="x", length=0, pad=4)
    fig.text(.128, .416, "shRNA knockdown", va="center")

    ax = fig.add_subplot(grid[1, 1])
    effect = crispri.loc[["GenomeWideScreen", "TargetedScreen"], "lfc"].to_numpy()
    ax.axvline(0, color=LIGHT, lw=.75, ls="--", zorder=0)
    ax.hlines([1, 0], effect, 0, color="#C3D2EA", lw=1.15)
    ax.scatter(effect, [1, 0], color=BLUE, s=23, lw=0, zorder=3)
    ax.set(yticks=[1, 0], yticklabels=["Genome-wide", "Targeted"], ylim=(-.45, 1.45),
           xlim=(-.41, .045), xticks=[-.4, -.2, 0], xlabel="Published log fold change")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(.614, .416, "CRISPRi", va="center")

    handles = [Line2D([], [], marker="o", color=GREY, lw=0, markersize=4, label="Control"),
               Line2D([], [], marker="D", color=SHRNA, lw=0, markersize=4, label="Knockdown")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005), frameon=False,
               ncol=2, handletextpad=.35, columnspacing=1.1)
    export(fig, OUT, "Figure_5_independent_TDGF1_RARRES2")


def observed_holdout_figure():
    from plot_idop_aligned_figures import CASES, collect_observed_holdout, draw_observed_holdout
    from screen_idop_transcriptome_readouts import load_states

    data_dir = RUN / "transcriptome_readout"
    protocol = json.loads((data_dir / "protocol.json").read_text(encoding="utf-8"))
    core, targets = protocol["core"], protocol["targets"]
    states, readouts, _ = load_states(core, targets, STAGE / "observed_readout_index.csv")
    genes = np.r_[core, readouts]
    np.testing.assert_array_equal(genes, np.load(data_dir / "nodes.npy"))
    assert states.shape == (32, 4, 15598)
    output = collect_observed_holdout(states, genes, targets)
    fig, axes = plt.subplots(1, len(CASES), sharey=True, figsize=(180 / 25.4, 2.38))
    fig.subplots_adjust(left=.115, right=.972, bottom=.28, top=.705, wspace=.24)
    handles = draw_observed_holdout(axes, output)
    fig.text(.440, .950, "PRDM14", ha="center", va="center", fontsize=9, fontweight="bold")
    fig.text(.902, .950, "POU5F1", ha="center", va="center", fontsize=9, fontweight="bold")
    fig.text(.545, .135, "Experimental day", ha="center")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.545, .005), frameon=False, ncol=2,
               handletextpad=.40, columnspacing=1.1)
    assert len(output) == 40 and output.observed_contrast.lt(0).all()
    output.to_csv(STAGE / "heldout_observed_trajectories.csv", index=False)
    export(fig, OUT, "Supplementary_observed_holdout_trajectories")


def read_profiles():
    full = pd.read_csv(RUN / "transcriptome_readout/all_scored_pairs.csv",
                       usecols=["source", "marker", "day5_response"])
    held = pd.read_csv(RUN / "manuscript_figures/Figure_3_genomewide_heldout_scores.csv",
                       usecols=["source", "readout", "heldout_response"])
    observed = pd.concat([
        chunk.loc[chunk.Target.isin(SOURCES)]
        for chunk in pd.read_csv(FENG, sep="\t", usecols=["Target", "Expressed_Gene_Symbol", "lfc"],
                                 chunksize=300_000)
    ]).dropna().drop_duplicates()
    result = {}
    for source in SOURCES:
        model = full.loc[full.source.eq(source)].set_index("marker").day5_response.rename("Full fit")
        refit = held.loc[held.source.eq(source)].set_index("readout").heldout_response.rename("Target holdout")
        assert model.index.is_unique and refit.index.is_unique and set(model.index) == set(refit.index)
        assert len(model) == 15575
        assay = observed.loc[observed.Target.eq(source)].set_index("Expressed_Gene_Symbol").lfc
        assay = assay.loc[~assay.index.duplicated(keep=False)].rename("External CRISPRi")
        assert len(assay) > 1000
        model_pair = pd.concat([model, refit], axis=1)
        common = model_pair.join(assay, how="inner")
        assert np.isfinite(model_pair.to_numpy()).all() and np.isfinite(common.to_numpy()).all()
        result[source] = (model_pair, common)
    return result


def hallmark_screen():
    terms = {}
    for line in GMT.read_text(encoding="utf-8").splitlines():
        name, _, *members = line.split("\t")
        terms[name] = {gene.strip().upper() for gene in members if gene.strip()}
    assert len(terms) == 50 and min(map(len, terms.values())) >= 15
    records = []
    for source, (model, common) in read_profiles().items():
        for universe, profiles in (("All model readouts", model), ("Shared with CRISPRi", common)):
            labels = profiles.index.str.upper()
            for profile, values in profiles.items():
                tests = []
                for term, genes in terms.items():
                    mask = labels.isin(genes)
                    n_members = int(mask.sum())
                    if n_members < 15:
                        continue
                    u, p = mannwhitneyu(values[mask], values[~mask], alternative="two-sided", method="asymptotic")
                    effect = 2 * u / (n_members * (len(values) - n_members)) - 1
                    tests.append(dict(source=source, universe=universe, profile=profile, hallmark=term,
                                      n_genes=len(values), n_members=n_members, rank_biserial=effect, p=p))
                for record, q in zip(tests, false_discovery_control([test["p"] for test in tests], method="bh")):
                    records.append(dict(record, q=q))
    result = pd.DataFrame(records)
    result.to_csv(STAGE / "hallmark_screen.csv", index=False)
    meta = dict(library="MSigDB Hallmark 2020 via Enrichr", gmt_sha256=hashlib.sha256(GMT.read_bytes()).hexdigest(),
                test="Two-sided Mann–Whitney with ties and BH within source, universe and profile",
                minimum_members=15, exploratory_q_threshold=SCREEN_Q,
                criterion="q <= 0.10 and same signed rank-biserial effect in full and target holdout; "
                          "external support requires same sign and q <= 0.10 in all three shared-universe profiles")
    (STAGE / "hallmark_protocol.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return result


def concordant_model_terms(effect, q):
    same_sign = np.sign(effect["Full fit"]) == np.sign(effect["Target holdout"])
    return same_sign & q["Full fit"].le(SCREEN_Q) & q["Target holdout"].le(SCREEN_Q)


def figure6(results):
    highlight = "#694083"
    fig = plt.figure(figsize=(180 / 25.4, 2.62))
    limits = (-.285, .356)
    whole = results.loc[results.universe.eq("All model readouts")]
    for i, (source, color) in enumerate(zip(SOURCES, ("#6F8FBE", "#BE8875"))):
        ax = fig.add_axes([.085 + .281 * i, .215, .218, .60])
        rows = whole.loc[whole.source.eq(source)]
        effect = rows.pivot(index="hallmark", columns="profile", values="rank_biserial")
        q = rows.pivot(index="hallmark", columns="profile", values="q")
        assert len(effect) == 50 and rows.n_genes.eq(15575).all()
        passing = concordant_model_terms(effect, q)
        assert list(effect.index[passing]) == (["Myc Targets V1"] if source == "PRDM14" else [])
        ax.plot(limits, limits, color="#CED6E0", lw=.8, zorder=0)
        ax.axhline(0, color=LIGHT, lw=.65, ls=":", zorder=0)
        ax.axvline(0, color=LIGHT, lw=.65, ls=":", zorder=0)
        ax.scatter(effect["Full fit"], effect["Target holdout"], color=color, alpha=.80,
                   edgecolors="white", linewidths=.35, s=19)
        for term, row in effect.loc[passing].iterrows():
            xy = (row["Full fit"], row["Target holdout"])
            ax.scatter(*xy, color=highlight, s=35, edgecolors="white", linewidths=.45, zorder=3)
            ax.annotate(term.replace("Myc", "MYC"), xy, xytext=(3, -14), textcoords="offset points",
                        va="center", color=highlight, fontsize=7.5,
                        arrowprops={"arrowstyle": "-", "color": highlight, "lw": .6})
        ax.set(xlim=limits, ylim=limits, xticks=[-.2, 0, .2], yticks=[-.2, 0, .2],
               xlabel="Full-fit enrichment", title=source)
        ax.set_aspect("equal", adjustable="box")
        if i == 0:
            ax.set_ylabel("Holdout enrichment")

    ax = fig.add_axes([.734, .215, .237, .60])
    rows = results.query("source == 'PRDM14' and universe == 'Shared with CRISPRi' and hallmark == 'Myc Targets V1'")
    order = ["Full fit", "Target holdout", "External CRISPRi"]
    rows = rows.set_index("profile").loc[order]
    assert rows.n_genes.eq(6422).all() and rows.n_members.eq(191).all()
    assert list(rows.q.le(SCREEN_Q)) == [True, False, True]
    ax.axvline(0, color=LIGHT, lw=.75, ls=":", zorder=0)
    for y, row, color, line_color in zip((2, 1, 0), rows.itertuples(), (BLUE, BLUE, SHRNA),
                                         ("#C3D2EA", "#C3D2EA", "#E6BAAA")):
        passed = row.q <= SCREEN_Q
        ax.hlines(y, row.rank_biserial, 0, color=line_color, lw=1.15)
        ax.scatter(row.rank_biserial, y, s=32, facecolors=color if passed else "white",
                   edgecolors=color, lw=1.05, zorder=3)
        ax.annotate(f"$q$ = {row.q:.3f}", (row.rank_biserial, y), xytext=(4, 6),
                    textcoords="offset points", color=color, fontsize=7.5)
    ax.set(yticks=[2, 1, 0], yticklabels=order, xlim=(-.183, .020), ylim=(-.52, 2.56),
           xticks=[-.15, -.10, -.05, 0], xlabel="Signed enrichment")
    ax.set_title("MYC Targets V1\n6,422 shared genes", fontsize=8, pad=4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    for x, letter in ((.030, "a"), (.335, "b"), (.628, "c")):
        fig.text(x, .891, letter, fontsize=9, fontweight="bold", va="center")
    export(fig, OUT, "Figure_6_Hallmark_enrichment_exploratory")


def compare_hallmarks(results):
    for source in SOURCES:
        print(source, flush=True)
        for universe in results.universe.unique():
            rows = results.loc[results.source.eq(source) & results.universe.eq(universe)]
            effect = rows.pivot(index="hallmark", columns="profile", values="rank_biserial")
            q = rows.pivot(index="hallmark", columns="profile", values="q")
            model_pass = concordant_model_terms(effect, q)
            summary = effect.loc[model_pass].join(q.add_prefix("q "))
            if "External CRISPRi" in effect:
                summary["external_support"] = ((np.sign(effect["Full fit"]) == np.sign(effect["External CRISPRi"]))
                                               & q["External CRISPRi"].le(SCREEN_Q)).loc[model_pass]
            print(universe, int(rows.n_genes.iloc[0]), "model concordant", int(model_pass.sum()), flush=True)
            print(summary.round(4).to_string(), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    figure5()
    results = hallmark_screen()
    compare_hallmarks(results)
    figure6(results)
    observed_holdout_figure()


if __name__ == "__main__":
    main()
