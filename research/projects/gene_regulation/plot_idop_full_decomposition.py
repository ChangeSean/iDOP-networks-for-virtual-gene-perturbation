"""Draw the full transcriptome network and cumulative response from the same fit."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.sparse import bmat, csr_matrix, eye, load_npz
from scipy.sparse.linalg import expm_multiply

from plot_idop_nebula import BLUE, CORE, DATA, INK, OUT, cloud_positions, draw_edges


NAME = "Figure_S4_full_idop_with_decomposition"
PURPLE, RUST, PALE = "#643982", "#AB422A", "#A7B4C5"
CASES = (("PRDM14", "DNMT3B"), ("PRDM14", "TDGF1"), ("PRDM14", "ESRG"),
         ("PRDM14", "MT1X"), ("POU5F1", "RARRES2"))


def decomposition(nodes, coupling, source_name, target_name, expected):
    source = np.flatnonzero(nodes == source_name).item()
    target = np.flatnonzero(nodes == target_name).item()
    assert source < CORE <= target and coupling[:CORE, CORE:].nnz == 0
    # The readout block is diagonal, so all upstream dependencies are retained here.
    idx = np.r_[np.arange(CORE), target]
    matrix = coupling[idx][:, idx]
    assert coupling[target, CORE:].nnz == 1
    p = len(idx)
    input_term = csr_matrix((-np.ones(1), ([source], [0])), shape=(p, 1))
    generator = bmat([[matrix, input_term, None],
                      [None, csr_matrix((1, 1)), None],
                      [eye(p), None, csr_matrix((p, p))]], format="csr")
    initial = np.zeros(2 * p + 1)
    initial[p] = 1.
    t = np.linspace(2., 5., 121)
    states = expm_multiply(generator, initial, start=0., stop=3., num=len(t))
    weights = matrix[-1].toarray().ravel()
    integrated = states[:, p + 1:] * weights
    total = states[:, p - 1]
    np.testing.assert_allclose(integrated.sum(axis=1), total, atol=2e-12, rtol=1e-11)
    np.testing.assert_allclose(total[-1], expected, atol=1e-12)
    data = {"day": t, "total": total, f"self_{target_name}": integrated[:, -1]}
    for i in np.flatnonzero(weights[:CORE]):
        data[nodes[i]] = integrated[:, i]
    return pd.DataFrame(data)


def network(ax, nodes, coupling, dnmt_inputs):
    coords, dominant, radius = cloud_positions(coupling, nodes[:CORE], nodes[CORE:])
    low = (coords[:CORE] - radius[:, None]).min(axis=0)
    high = (coords[:CORE] + radius[:, None]).max(axis=0)
    ax.set_aspect("equal")
    ax.set(xlim=(low[0] - .28, high[0] + .33), ylim=(low[1] - .15, high[1] + .09))
    ax.set_axis_off()
    draw_edges(ax, coupling, coords, dominant)
    degree = coupling[CORE:, :CORE].getnnz(axis=1)
    ax.scatter(*coords[CORE:].T, s=.17 + .095 * degree, color="#294C78",
               alpha=.43, linewidths=0, zorder=3)
    ax.scatter(*coords[:CORE].T, s=14, facecolor="#F4F7FB", edgecolor="#274467",
               linewidths=.61, zorder=4)
    index = {name: i for i, name in enumerate(nodes)}
    for sender in dnmt_inputs.drop("PRDM14"):
        i, j = index[sender], index["DNMT3B"]
        color = BLUE if coupling[j, i] > 0 else RUST
        ax.add_patch(FancyArrowPatch(coords[i], coords[j], arrowstyle="-|>",
                                     connectionstyle="arc3,rad=-.085", shrinkA=3, shrinkB=4,
                                     mutation_scale=4.8, color=color, lw=.66, alpha=.65, zorder=3.5))

    # Each emphasized arrow corresponds to an existing entry in the coupling matrix.
    for (sender, receiver), bend in zip(CASES, (.13, -.14, .20, .16, -.55)):
        i, j = index[sender], index[receiver]
        assert coupling[j, i] > 0
        for color, width, layer in [("white", 2.45, 5), (BLUE, 1.23, 6)]:
            ax.add_patch(FancyArrowPatch(coords[i], coords[j], arrowstyle="-|>",
                                         connectionstyle=f"arc3,rad={bend}", shrinkA=2.9, shrinkB=2.5,
                                         mutation_scale=6.3, color=color, lw=width, zorder=layer))

    focus_offsets = {"PRDM14": (0, 9), "POU5F1": (-8, 9), "DNMT3B": (7, 8),
                     "TDGF1": (2, 10), "ESRG": (8, -8), "MT1X": (-4, 10), "RARRES2": (0, -9)}
    input_offsets = {"ETV4": (5, -9), "FOXH1": (1, -9), "JARID2": (-9, -4), "MYC": (3, 8),
                     "MYCN": (-8, 8), "NANOG": (-7, 10), "RUNX1T1": (6, -8), "SOX2": (0, 8),
                     "TRIM25": (9, -8), "VENTX": (-7, -9)}
    assert set(input_offsets) == set(dnmt_inputs) - {"PRDM14"}
    for offsets, primary in ((input_offsets, False), (focus_offsets, True)):
        for gene, (dx, dy) in offsets.items():
            xy = coords[index[gene]]
            if primary:
                ax.scatter(*xy, s=22 if gene in {"PRDM14", "POU5F1"} else 15,
                           color=BLUE, edgecolors="white", linewidths=.7, zorder=7)
            ax.annotate(gene, xy, xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx >= 5 else "right" if dx <= -5 else "center", va="center",
                        fontsize=6.5 if primary else 5.6, color=BLUE if primary else INK,
                        weight="bold" if primary else "normal",
                        bbox=dict(fc="white", ec="none", pad=.55, alpha=.95),
                        arrowprops=dict(arrowstyle="-", color=BLUE if primary else "#6C7C8D",
                                        lw=.5, shrinkA=1, shrinkB=3), zorder=8)


def focal_network(ax, nodes, coupling):
    index = {name: i for i, name in enumerate(nodes)}
    sources = {"PRDM14": .64, "POU5F1": .15}
    receivers = (.91, .73, .55, .37, .15)
    weights = [float(coupling[index[target], index[source]]) for source, target in CASES]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    for source, y in sources.items():
        ax.scatter(.34, y, s=25, color=BLUE, zorder=3)
        ax.text(.30, y, source, ha="right", va="center", fontsize=7, color=INK, weight="bold")
    for (source, target), y, weight in zip(CASES, receivers, weights):
        ax.add_patch(FancyArrowPatch((.34, sources[source]), (.70, y), arrowstyle="-|>",
                                     mutation_scale=7, shrinkA=4, shrinkB=4,
                                     color=BLUE, lw=.45 + 1.1 * weight / max(weights)))
        ax.scatter(.70, y, s=20, facecolor="white", edgecolor=BLUE, linewidth=.9, zorder=3)
        ax.text(.75, y, target, ha="left", va="center", fontsize=7, color=INK)


def response_panel(ax, frame, target, main_input, ylim, show_y):
    elapsed = frame.day - frame.day.iloc[0]
    ax.axhline(0., color=PALE, lw=.55, zorder=0)
    for gene in frame.columns[3:].drop(main_input):
        ax.plot(elapsed, frame[gene], color="#929FB1", lw=.57, alpha=.72, zorder=1,
                gid=f"input-{target}-{gene}")
    lines = ((f"self_{target}", "Self", RUST, 1.05, "-"),
             ("total", "Total", BLUE, 1.28, "-"),
             (main_input, main_input, PURPLE, 1.10, "--"))
    for column, label, color, width, linestyle in lines:
        ax.plot(elapsed, frame[column], color=color, lw=width, ls=linestyle,
                gid=f"curve-{target}-{column}")
        ax.annotate(label, (elapsed.iloc[-1], frame[column].iloc[-1]), xytext=(2.4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=5.8,
                    color=color, annotation_clip=False)
    ax.set(xlim=(-.02, 4.25), ylim=ylim, xticks=[0, 1, 2, 3])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.4, width=.6, pad=3, labelleft=show_y)
    ax.set_title(target, loc="left", fontsize=7, color=INK, weight="bold", pad=5)


def main():
    nodes = np.load(DATA / "nodes.npy", allow_pickle=False)
    coupling = load_npz(DATA / "coupling.npz").tocsr()
    assert len(nodes) == 15598 and coupling.shape == (len(nodes), len(nodes))
    assert coupling.nnz - np.count_nonzero(coupling.diagonal()) == 115369
    archived = pd.read_csv(DATA / "all_scored_pairs.csv").set_index(["source", "marker"])
    frames = {(source, target): decomposition(nodes, coupling, source, target,
                                               archived.loc[(source, target), "day5_response"])
              for source, target in CASES}
    frame = frames[CASES[0]]
    plt.rcParams.update({"font.size": 7, "axes.linewidth": .65,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig = plt.figure(figsize=(180 / 25.4, 122 / 25.4), facecolor="white")
    network(fig.add_axes([.007, .341, .645, .642]), nodes, coupling, frame.columns[3:])
    focal_network(fig.add_axes([.630, .385, .336, .554]), nodes, coupling)
    fig.text(.025, .958, "a", weight="bold", fontsize=8, color=INK)
    fig.text(.660, .958, "b", weight="bold", fontsize=8, color=INK)
    main_inputs = {pair: f.iloc[-1, 3:].abs().idxmax() for pair, f in frames.items()}
    visible = np.concatenate([f.iloc[:, 1:].to_numpy().ravel() for f in frames.values()])
    ymin, ymax = visible.min(), visible.max()
    margin = .055 * (ymax - ymin)
    for i, (source, target) in enumerate(CASES):
        x = .069 + i * .183
        fig.text(x - .027, .303, "cdefg"[i], weight="bold", fontsize=8, color=INK)
        ax = fig.add_axes([x, .083, .177, .207])
        response_panel(ax, frames[source, target], target, main_inputs[source, target],
                       (ymin - margin, ymax + margin), i == 0)
    fig.text(.515, .018, "Elapsed simulation time (days)", ha="center", fontsize=7, color=INK)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{NAME}.{extension}", dpi=350)
    plt.close(fig)
    frame.to_csv(OUT / "Figure_S4_DNMT3B_decomposition_curves.csv", index=False)
    data = pd.concat([f.melt(id_vars="day", var_name="term", value_name="contribution")
                      .assign(source=source, readout=target) for (source, target), f in frames.items()],
                     ignore_index=True)
    data.to_csv(OUT / "Figure_S4_five_readout_decompositions.csv", index=False)
    print({target: main_inputs[source, target] for source, target in CASES})


if __name__ == "__main__":
    main()
