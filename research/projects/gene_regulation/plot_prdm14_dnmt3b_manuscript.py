"""Build the manuscript figures and source tables from the frozen feasibility run."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.text import Text
import networkx as nx
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from idop_diffusion import TimeSeriesBatch
from idop_diffusion.propagation import LinearPropagation

RESULTS = ROOT / "results/idop_feasibility_20261109"
REAL = RESULTS / "real"
JOINT = RESULTS / "joint_biology"
OUT = RESULTS / "manuscript_figures"

INK = "#202631"
MUTED = "#555e6c"
GRID = "#c5cfd3"
DOWN = "#3156a3"
BLUE = "#0072b2"
UP = "#d55e00"
CONTROL = "#7a7f89"
PALE = "#eef1f8"
JITTER = [-.085, 0, .085]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def heading(ax, panel, title):
    ax.annotate(panel.lower(), (0, 1), xycoords="axes fraction", xytext=(-14, 10),
                textcoords="offset points", fontsize=10, weight="bold",
                va="baseline", gid="panel-label")
    ax.set_title(title, loc="left", fontsize=7, fontweight="normal", pad=10)


def blank(ax):
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))


def reference(ax, axis="y"):
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(0, color=CONTROL, lw=.8, ls=(0, (3, 3)), zorder=0)


def save(fig, name):
    fig.set_size_inches(180 / 25.4, fig.get_figheight())
    for label in fig.findobj(Text):
        if label.get_gid() == "dense-label":
            label.set_fontsize(7)
        elif label.get_gid() != "panel-label":
            label.set_fontsize(8)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def arrow(ax, start, end, color=CONTROL, shrink=0):
    ax.annotate("", xy=end, xytext=start, arrowprops={
        "arrowstyle": "-|>", "color": color, "lw": 1.1,
        "mutation_scale": 9, "shrinkA": shrink, "shrinkB": shrink,
    })


def box(ax, x, y, w, h, title, detail, face="#f2f5f6", title_size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
                                ec="none", fc=face))
    ax.text(x + w / 2, y + h * .69, title, ha="center", va="center",
            fontsize=title_size, weight="bold")
    ax.text(x + w / 2, y + h * .32, detail, ha="center", va="center",
            fontsize=7, color=MUTED, linespacing=1.24)


def external_crispr():
    panel = ROOT / "data/GSE303158/primed_panel"
    meta = pd.read_csv(panel / "cell_metadata.csv.gz", usecols=[
        "barcode", "target", "batch", "feature_call", "num_features", "nCount_RNA",
    ])
    guides = json.loads((JOINT / "frozen_external_guides.json").read_text())["PRDM14"]
    assert meta.num_features.eq(1).all() and meta.barcode.is_unique
    with np.load(panel / "counts.npz") as data:
        np.testing.assert_array_equal(data["barcodes"], meta.barcode)
        np.testing.assert_array_equal(data["library_size"], meta.nCount_RNA)
        gene = data["markers"].tolist().index("DNMT3B")
        values = np.log1p(data["counts"][:, gene] * (1e4 / data["library_size"]))

    rows = []
    for guide in guides:
        for batch in sorted(meta.batch.unique()):
            matched = meta.batch.eq(batch)
            ctrl = values[matched & meta.target.eq("CTRL")]
            pert = values[matched & meta.feature_call.eq(guide)]
            delta = pert.mean() - ctrl.mean()
            se = np.sqrt(pert.var(ddof=1) / len(pert) + ctrl.var(ddof=1) / len(ctrl))
            rows.append(dict(guide=guide, batch=batch, guide_cells=len(pert),
                             non_targeting_cells=len(ctrl), mean_guide_log1p_cp10k=pert.mean(),
                             mean_control_log1p_cp10k=ctrl.mean(), delta_log1p_cp10k=delta,
                             welch_t=delta / se))
    result = pd.DataFrame(rows)
    frozen = pd.read_csv(JOINT / "replicated_biology_candidates.csv").set_index("source").loc["PRDM14"]
    delta = result.delta_log1p_cp10k
    np.testing.assert_allclose([delta.min(), delta.max(), delta.mean()], frozen[
        ["external_min_delta", "external_max_delta", "external_mean_delta"]].astype(float),
        atol=1e-12)
    np.testing.assert_allclose(-result.groupby("guide").welch_t.mean().max(),
                               frozen.external_min_directed_guide_t, atol=1e-12)
    assert delta.lt(0).all() and len(result) == 4
    result.to_csv(OUT / "Table_S2A_primed_CRISPR.csv", index=False)
    return result


def group_points(ax, groups, treatment_color):
    for x, (values, color, marker) in enumerate(zip(groups, [CONTROL, treatment_color], ["o", "D"])):
        ax.scatter(x + np.array(JITTER), values, c=color, marker=marker, s=20, zorder=3,
                   linewidths=.45, edgecolors="white")
        ax.hlines(np.mean(values), x - .22, x + .22, colors=color, lw=1.25, zorder=2)
    ax.set_xlim(-.42, 1.42)
    ax.tick_params(axis="x", length=0)


def figure_3():
    shrna_color = "#B45337"
    depletion_color = "#76518D"
    crispr = external_crispr()
    probes = pd.read_csv(JOINT / "prdm14_shrna_probe_check.csv").query("gene == 'DNMT3B'")
    probes.to_csv(OUT / "Table_S2B_shRNA_probes.csv", index=False)
    regions = pd.read_csv(ROOT / "data/GSE164786/DNMT3B_regions.csv")
    control_cols = regions.filter(regex=r"_No_IAA_Rep[1-3]_RNAseq$").columns
    treated_cols = regions.filter(regex=r"_Plus_IAA_Rep[1-3]_RNAseq$").columns
    diffs = regions[treated_cols].mean(axis=1) - regions[control_cols].mean(axis=1)
    separation = regions[treated_cols].max(axis=1) < regions[control_cols].min(axis=1)
    np.testing.assert_allclose(diffs, regions.IAA_minus_control_mean_log2RPM)
    assert (len(regions), diffs.lt(0).sum(), separation.sum()) == (29, 29, 28)
    assert len(probes) == 5 and probes.all_knockdown_below_controls.sum() == 4
    regions.to_csv(OUT / "Table_S2C_naive_RNA_regions.csv", index=False)

    fig = plt.figure(figsize=(180 / 25.4, 5.45))
    layout = fig.add_gridspec(3, 1, left=.14, right=.95, top=.93, bottom=.105,
                              height_ratios=[.9, 1.05, 1.18], hspace=.83)
    ax = fig.add_subplot(layout[0])
    heading(ax, "A", "CRISPR")
    reference(ax, "x")
    for batch, offset, marker, color in zip(sorted(crispr.batch.unique()), [.13, -.13],
                                            ["o", "s"], [DOWN, "#5987BB"]):
        group = crispr.loc[crispr.batch.eq(batch)]
        y = np.array([1, 0]) + offset
        ax.scatter(group.delta_log1p_cp10k, y, color=color, s=29, marker=marker,
                   label=f"Batch {batch}")
    ax.set(yticks=[1, 0], yticklabels=["sg_7", "sg_8"], ylim=(-.43, 1.5), xlim=(-.60, .24),
           xlabel="DNMT3B contrast (log1p CP10k)")
    ax.set_xticks([-.6, -.4, -.2, 0])
    ax.legend(loc="upper right", fontsize=7, borderaxespad=.15, labelspacing=.2)

    b = fig.add_subplot(layout[1])
    blank(b)
    heading(b, "B", "shRNA knockdown")
    probe_grid = layout[1].subgridspec(1, 5, wspace=.62)
    sample_cols = probes.filter(regex=r"^GSM\d+$").columns
    for idx, row in enumerate(probes.itertuples(index=False)):
        ax = fig.add_subplot(probe_grid[0, idx])
        ref = np.array([getattr(row, col) for col in sample_cols[:3]])
        kd = np.array([getattr(row, col) for col in sample_cols[3:]])
        reference(ax)
        group_points(ax, [ref - ref.mean(), kd - ref.mean()], shrna_color)
        ax.set_xticks([])
        ax.set_xlabel(f"Probe {idx + 1}", labelpad=3)
        if idx == 0:
            ax.set_ylabel("Centered signal")
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=7)

    c = fig.add_subplot(layout[2])
    blank(c)
    heading(c, "C", "Auxin-induced depletion")
    c_grid = layout[2].subgridspec(1, 2, width_ratios=[1, 1.45], wspace=.47)
    ax = fig.add_subplot(c_grid[0, 0])
    group_points(ax, [regions[control_cols].median().to_numpy(), regions[treated_cols].median().to_numpy()],
                 depletion_color)
    ax.set(xticks=[0, 1], xticklabels=["Control", "Auxin"], ylabel="Median signal (log2 RPM)",
           ylim=(5.1, 7.45))
    ax = fig.add_subplot(c_grid[0, 1])
    ordering = np.argsort(diffs.to_numpy())
    ax.scatter(np.arange(1, 30), diffs.iloc[ordering], s=15, color=depletion_color, linewidths=0)
    reference(ax)
    ax.set(xlabel="Ranked regions", ylabel="Contrast (log2 RPM)",
           xticks=[1, 15, 29], xlim=(-.5, 30.5), ylim=(-1.42, .12))
    save(fig, "Figure_3_independent_human_evidence")


def trajectory():
    model = LinearPropagation.load(JOINT / "model/propagation.npz")
    source = model.nodes.index("PRDM14")
    target = model.nodes.index("DNMT3B")
    days = np.arange(2, 6)
    observed = []
    for guide in (1, 2):
        batch = TimeSeriesBatch.load(REAL / f"reference_guide{guide}/training.npz")
        row, = np.flatnonzero(batch.source_indices == batch.nodes.index("PRDM14"))
        np.testing.assert_array_equal(batch.times, days)
        observed.append(batch.states[row, :, batch.nodes.index("DNMT3B")])
    virtual = [-model.operators(day - 2)[1][target, source] for day in days]
    frame = pd.DataFrame(dict(day=days, virtual_response=virtual,
                              measured_guide_1=observed[0], measured_guide_2=observed[1]))
    frame.to_csv(OUT / "Figure_2_trajectory_values.csv", index=False)
    dense_days = np.linspace(2, 5, 49)
    dense_response = [-model.operators(day - 2)[1][target, source] for day in dense_days]
    return frame, dense_days, dense_response


def signed_edge(ax, a, b, weight, width, shrink=14, alpha=1, curve=0, patches=(None, None)):
    patch = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7,
                            linewidth=width, color=BLUE if weight > 0 else UP,
                            linestyle="-" if weight > 0 else (0, (2.3, 1.5)),
                            shrinkA=shrink, shrinkB=shrink, alpha=alpha, patchA=patches[0], patchB=patches[1],
                            connectionstyle=f"arc3,rad={curve}", zorder=1)
    ax.add_patch(patch)


def gene_label(ax, xy, gene, highlight=False):
    return ax.text(*xy, gene, ha="center", va="center", fontsize=8,
            weight="bold" if highlight else "normal", zorder=4,
            bbox=dict(boxstyle="round,pad=.27", fc="#e7f2f7" if highlight else "white",
                      ec=BLUE if highlight else GRID, lw=.65))


def local_network(ax):
    blank(ax)
    heading(ax, "a", "Local structure around PRDM14 and DNMT3B")
    all_edges = pd.read_csv(JOINT / "model/propagation_parameters.csv").query("role == 'cross_node'")
    edges = all_edges.loc[all_edges.source.eq("PRDM14") | all_edges.target.eq("DNMT3B")].copy()
    edges.to_csv(OUT / "Figure_2_local_network_edges.csv", index=False)
    targets = sorted(set(edges.loc[edges.source.eq("PRDM14"), "target"]) - {"DNMT3B"})
    parents = sorted(set(edges.loc[edges.target.eq("DNMT3B"), "source"]) - {"PRDM14"})
    pos = {"PRDM14": (.36, .49), "DNMT3B": (.66, .49)}
    pos.update({gene: (.075, y) for gene, y in zip(targets, np.linspace(.09, .89, len(targets)))})
    pos.update({gene: (.935, y) for gene, y in zip(parents, np.linspace(.13, .85, len(parents)))})
    labels = {gene: gene_label(ax, xy, gene, gene in ("PRDM14", "DNMT3B"))
              for gene, xy in pos.items()}
    ax.figure.canvas.draw()
    scale = edges.propagation_coefficient.abs().max()
    for row in edges.itertuples():
        start = np.array(pos[row.source])
        end = np.array(pos[row.target])
        vector = end - start
        trim = 1 / max(abs(vector[0]) / .061, abs(vector[1]) / .026)
        signed_edge(ax, start + trim * vector, end - trim * vector,
                    row.propagation_coefficient,
                    .4 + 1.55 * abs(row.propagation_coefficient) / scale, shrink=0)
    ax.text(.51, .555, "+0.260", ha="center", fontsize=6.5, color=BLUE)
    ax.text(.36, .35, "Negative input", ha="center", fontsize=6, color=MUTED)
    ax.text(.66, .35, "Nominated decrease", ha="center", fontsize=6, color=DOWN)
    ax.text(.075, .965, "PRDM14 outputs", ha="center", fontsize=6, color=MUTED)
    ax.text(.935, .965, "DNMT3B inputs", ha="center", fontsize=6, color=MUTED)
    signed_edge(ax, (.32, .17), (.43, .17), 1, 1.1, shrink=0)
    signed_edge(ax, (.53, .17), (.64, .17), -1, 1.1, shrink=0)
    ax.text(.375, .11, "Positive", ha="center", fontsize=6)
    ax.text(.585, .11, "Negative", ha="center", fontsize=6)
    ax.text(.50, .015, "Width scales with |coupling|; all incident edges shown",
            ha="center", fontsize=6, color=MUTED)


def whole_network():
    model = LinearPropagation.load(JOINT / "model/propagation.npz")
    edges = pd.read_csv(JOINT / "model/propagation_parameters.csv").query("role == 'cross_node'")
    graph = nx.DiGraph()
    graph.add_nodes_from(model.nodes)
    graph.add_edges_from(zip(edges.source, edges.target))
    positions = nx.circular_layout(graph)
    fig = plt.figure(figsize=(180 / 25.4, 6.8))
    ax = fig.add_axes([.11, .16, .78, .74])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set(xlim=(-1.23, 1.23), ylim=(-1.23, 1.23))
    heading(ax, "a", "Joint inferred network · 51 genes, 458 directed edges")
    focal = edges.source.eq("PRDM14") | edges.target.eq("DNMT3B")
    for subset, alpha, width in [(edges.loc[~focal], .11, .35), (edges.loc[focal], .85, .8)]:
        for row in subset.itertuples():
            signed_edge(ax, positions[row.source], positions[row.target],
                        row.propagation_coefficient, width, shrink=3, alpha=alpha, curve=.06)
    for gene, xy in positions.items():
        highlight = gene in ("PRDM14", "DNMT3B")
        ax.scatter(*xy, s=26 if highlight else 9, color=BLUE if highlight else CONTROL,
                   zorder=3, edgecolor="white", linewidth=.4)
        label_xy = xy * 1.06
        angle = np.degrees(np.arctan2(xy[1], xy[0]))
        rotate = angle if -90 <= angle <= 90 else angle + 180
        ax.text(*label_xy, gene, fontsize=5.5, rotation=rotate, rotation_mode="anchor",
                ha="left" if xy[0] >= 0 else "right", va="center", gid="dense-label",
                weight="bold" if highlight else "normal")
    fig.text(.5, .085, "Highlighted edges match Supplementary Fig. 4a.",
             ha="center", fontsize=7)
    fig.text(.5, .055, "Blue solid = positive   ·   Orange dashed = negative   ·   Arrows = fitted direction",
             ha="center", fontsize=6.5)
    save(fig, "Figure_S1_complete_inferred_network")


def figure_2():
    curves, dense_days, dense_response = trajectory()
    picks = pd.read_csv(JOINT / "frozen_biology_candidates.csv").sort_values("model_day5_standard_input")
    replicated = pd.read_csv(JOINT / "replicated_biology_candidates.csv")
    np.testing.assert_allclose(curves.virtual_response.iloc[-1],
                               picks.set_index("source").loc["PRDM14", "model_day5_standard_input"])

    fig = plt.figure(figsize=(180 / 25.4, 7.8))
    layout = fig.add_gridspec(3, 2, left=.115, right=.96, top=.94, bottom=.085,
                              height_ratios=[2.7, 1, 1.35], wspace=.48, hspace=.56)
    a = fig.add_subplot(layout[0, :])
    local_network(a)
    b = fig.add_subplot(layout[1, :])
    pos = b.get_position()
    b.set_position([pos.x0 + .067, pos.y0, pos.width - .067, pos.height])
    heading(b, "B", "Two nominated responses from four eligible sources")
    reference(b, "x")
    b.axvline(-.03, color=GRID, lw=.8, ls=":")
    for y, row in zip([1, 0], picks.itertuples()):
        passes = (replicated.source.eq(row.source) & replicated.marker.eq(row.marker)).any()
        color = DOWN if passes else CONTROL
        b.hlines(y, row.model_day5_standard_input, 0, color=color, alpha=.45, lw=1.2)
        b.scatter(row.model_day5_standard_input, y, color=color, s=33, zorder=3)
        b.text(.012, y, f"{round(row.bootstrap_sign_agreement * 32)}/32   "
               f"{'passed external screen' if passes else 'below external criteria'}",
               va="center", color=color, fontsize=7.5)
    b.set(yticks=[1, 0], yticklabels=[f"{r.source} → {r.marker}" for r in picks.itertuples()],
          ylim=(-.5, 1.52), xlim=(-.205, .15), xlabel="Virtual day-5 response (model units)")
    b.set_xticks([-.20, -.15, -.10, -.05, 0])

    c = fig.add_subplot(layout[2, 0])
    heading(c, "C", "Virtual DNMT3B response")
    reference(c)
    c.plot(dense_days, dense_response, color=DOWN, lw=1.5)
    c.scatter(curves.day, curves.virtual_response, s=21, color=DOWN, zorder=3)
    c.set(xticks=curves.day, xlabel="Day (model anchor at day 2)", ylabel="Response (model units)",
          ylim=(-.205, .025))
    c.text(4.98, -.179, f"{curves.virtual_response.iloc[-1]:.3f}", va="top", ha="right", color=DOWN)
    d = fig.add_subplot(layout[2, 1])
    heading(d, "D", "Measured DNMT3B contrast")
    reference(d)
    d.plot(curves.day, curves.measured_guide_1, "-o", color=DOWN, lw=1.5, ms=4.8, label="Guide 1")
    d.plot(curves.day, curves.measured_guide_2, "--s", color=BLUE, lw=1.25, ms=4.5, label="Guide 2")
    d.set(xticks=curves.day, xlabel="Days after transduction", ylabel="Mean log1p CP10k contrast",
          ylim=(-.9, .1))
    d.legend(loc="upper right", fontsize=7, labelspacing=.25, borderaxespad=.2,
             frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    save(fig, "Figure_2_candidate_nomination")


def figure_1():
    guides = json.loads((JOINT / "frozen_external_guides.json").read_text())
    with np.load(ROOT / "data/GSE303158/primed_panel/counts.npz") as data:
        marker_count = len(data["markers"])
    nominations = pd.read_csv(JOINT / "frozen_biology_candidates.csv")
    retained = pd.read_csv(JOINT / "replicated_biology_candidates.csv")

    fig, ax = plt.subplots(figsize=(180 / 25.4, 3.35))
    fig.subplots_adjust(left=.04, right=.96, bottom=.09, top=.94)
    blank(ax)
    stages = [
        ("a", "Fit the joint network", "GSE213069 · days 2–5\n16 perturbations, two guides\n51 genes · iDOP edge selection"),
        ("b", "Simulate gene inputs", f"{len(guides)} eligible perturbations × {marker_count} markers\n"
         f"{len(guides) * marker_count} source–marker pairs\nRank absolute day-5 response"),
        ("c", "Screen and freeze", "Absolute response ≥ 0.03\nSign agreement ≥ 80% of 32 refits\n"
         f"{len(nominations)} pairs nominated"),
        ("d", "Check independent data", f"{len(retained)} pair passed the CRISPR screen\n"
         "PRDM14–DNMT3B\nFollow-up in shRNA and naïve PSCs"),
    ]
    for i, (label, title, detail) in enumerate(stages):
        x = .015 + .505 * (i % 2)
        y = .57 if i < 2 else .03
        ax.text(x, y + .35, label, fontsize=10, weight="bold", gid="panel-label")
        ax.text(x + .04, y + .35, title, weight="bold")
        box(ax, x, y, .465, .28, detail.split("\n")[0], "\n".join(detail.split("\n")[1:]),
            face=PALE if i >= 2 else "#f2f5f6")
    arrow(ax, (.485, .705), (.515, .705))
    arrow(ax, (.485, .16), (.515, .16))
    ax.plot([.75, .75, .25], [.56, .47, .47], color=CONTROL, lw=1.1)
    arrow(ax, (.25, .47), (.25, .405))
    save(fig, "Figure_1_nomination_workflow")


def main():
    OUT.mkdir(exist_ok=True)
    figure_3()
    figure_2()
    figure_1()
    whole_network()


if __name__ == "__main__":
    main()

