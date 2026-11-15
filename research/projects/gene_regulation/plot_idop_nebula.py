"""Draw the fitted iDOP network as separated dominant-input gene clouds."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as ArtistPath
import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/idop_feasibility_20261109"
DATA = RUN / "transcriptome_readout"
OUT = RUN / "manuscript_figures"
CORE = 23
BLUE, ORANGE = "#2459a9", "#cf651f"
INK, MUTED = "#202631", "#555e6c"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "text.color": INK,
                     "pdf.fonttype": 42, "svg.fonttype": "none",
                     "savefig.facecolor": "white"})


def gravity_layout(weights, radius):
    """Pull connected groups together and enforce clearance between cloud envelopes."""
    rng = np.random.default_rng(20261109)
    centers = rng.normal(0., .75, size=(CORE, 2))
    minimum = radius[:, None] + radius[None, :] + .16
    pairs = tuple(zip(*np.triu_indices(CORE, k=1)))
    for step in range(850):
        offset = centers[None, :, :] - centers[:, None, :]
        distance = np.maximum(np.linalg.norm(offset, axis=2), 1e-8)
        direction = offset / distance[..., None]
        pull = .16 * weights * np.maximum(distance - minimum, 0)
        push = .046 / (distance**2 + .12)
        force = ((pull - push)[..., None] * direction).sum(axis=1) - .15 * centers
        centers += (.065 if step < 550 else .022) * force
        # Project onto the minimum-distance constraints after each force step.
        for _ in range(6):
            for i, j in pairs:
                difference = centers[j] - centers[i]
                separation = np.linalg.norm(difference)
                deficit = minimum[i, j] - separation
                if deficit > 0:
                    move = .5 * deficit * difference / separation
                    centers[i] -= move
                    centers[j] += move
    centers -= centers.mean(axis=0)
    # Resolve residual simultaneous contacts without changing the relative layout.
    i, j = np.triu_indices(CORE, k=1)
    separation = np.linalg.norm(centers[i] - centers[j], axis=1)
    centers *= max(1., np.max(minimum[i, j] / separation))
    # Orient the widest layout axis horizontally. This preserves pair distances.
    _, vectors = np.linalg.eigh(centers.T @ centers)
    return centers @ vectors[:, ::-1]


def cloud_positions(matrix, core_names, readout_names):
    readouts = matrix[CORE:, :CORE].tocsr()
    magnitudes = abs(readouts)
    dominant = np.asarray(magnitudes.argmax(axis=1)).ravel()
    counts = np.bincount(dominant, minlength=CORE)
    radius = .105 + .40 * np.sqrt(counts / counts.max())
    membership = np.eye(CORE)[dominant]
    cloud_strength = (membership.T @ magnitudes) / counts[:, None]
    core_strength = np.abs(matrix[:CORE, :CORE].toarray())
    strength = cloud_strength + core_strength
    weights = strength + strength.T
    np.fill_diagonal(weights, 0.)
    weights /= np.median(weights[weights > 0])
    centers = gravity_layout(weights, radius)
    core_index = {gene: i for i, gene in enumerate(core_names)}
    dnmt3b_group = dominant[readout_names.tolist().index("DNMT3B")]
    if centers[core_index["PRDM14"], 0] > centers[dnmt3b_group, 0]:
        centers[:, 0] *= -1

    rng = np.random.default_rng(20261109)
    theta = rng.uniform(0., 2 * np.pi, len(dominant))
    radial = .13 + .87 * rng.beta(1.65, 2.25, len(dominant))
    offsets = radial[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
    incoming_strength = np.asarray(magnitudes.sum(axis=1)).ravel()
    attraction = (magnitudes @ centers) / incoming_strength[:, None] - centers[dominant]
    norm = np.linalg.norm(attraction, axis=1)
    offsets += .25 * attraction / (.40 + norm[:, None])
    # Keep readouts inside the constrained envelope while retaining the direction of attraction.
    offsets /= np.maximum(1., np.linalg.norm(offsets, axis=1))[:, None]
    outputs = centers[dominant] + radius[dominant, None] * offsets
    return np.vstack((centers, outputs)), dominant, radius


def draw_edges(ax, matrix, coords, dominant):
    edge = matrix[:, :CORE].tocoo()
    cross_gene = edge.row != edge.col
    target, source, coefficient = edge.row[cross_gene], edge.col[cross_gene], edge.data[cross_gene]
    assert len(source) == 115369
    start, end = coords[source], coords[target]
    is_readout = target >= CORE
    receiver = target.copy()
    receiver[is_readout] = dominant[target[is_readout] - CORE]
    own_cloud = is_readout & (source == receiver)
    delta = coords[receiver] - start
    normal = np.column_stack((-delta[:, 1], delta[:, 0]))
    control1 = start + .34 * delta + .10 * normal
    control2 = start + .79 * delta + .10 * normal
    control1[own_cloud] = .72 * start[own_cloud] + .28 * end[own_cloud]
    control2[own_cloud] = .30 * start[own_cloud] + .70 * end[own_cloud]
    codes = [ArtistPath.MOVETO, ArtistPath.CURVE4, ArtistPath.CURVE4, ArtistPath.CURVE4]
    between = is_readout & ~own_cloud
    strong_between = np.zeros_like(between)
    for i in range(CORE):
        choices = np.flatnonzero(between & (source == i))
        ranked = choices[np.argsort(-np.abs(coefficient[choices]), kind="stable")]
        _, first_by_cloud = np.unique(receiver[ranked], return_index=True)
        strongest = ranked[np.sort(first_by_cloud)[:3]]
        strong_between[strongest] = True
    styles = [(between & ~strong_between, .0018, .19),
              (strong_between, .33, .48),
              (~is_readout, .21, .40), (own_cloud, .090, .28)]
    for group, alpha, width in styles:
        for sign, color in [(1, BLUE), (-1, ORANGE)]:
            show = group & (sign * coefficient > 0)
            vertices = np.stack((start[show], control1[show], control2[show], end[show]), axis=1)
            paths = [ArtistPath(points, codes) for points in vertices]
            ax.add_collection(PathCollection(paths, transform=ax.transData, facecolors="none",
                              edgecolors=[(*to_rgb(color), alpha)], linewidths=width, zorder=1))


def draw_focus(ax, nodes, matrix, coords):
    index = {gene: i for i, gene in enumerate(nodes)}
    start, end = index["PRDM14"], index["DNMT3B"]
    assert matrix[end, start] > 0
    for color, width, layer in [("white", 3.1, 5), (BLUE, 1.65, 6)]:
        ax.add_patch(FancyArrowPatch(coords[start], coords[end], arrowstyle="-|>", mutation_scale=8,
                                    linewidth=width, color=color, shrinkA=5, shrinkB=4,
                                    connectionstyle="arc3,rad=.16", zorder=layer))
    ax.scatter(*coords[[start, end]].T, s=[33, 25], color=BLUE, edgecolors="white",
               linewidths=.8, zorder=7)
    ax.annotate("DNMT3B", xy=coords[end], xytext=(17, 11), textcoords="offset points",
                fontsize=8, weight="bold", color=BLUE, ha="left", va="center",
                bbox=dict(fc="white", ec="none", boxstyle="round,pad=.16", alpha=.95),
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=.65, shrinkA=1, shrinkB=3), zorder=8)


def main():
    nodes = np.load(DATA / "nodes.npy", allow_pickle=False)
    matrix = load_npz(DATA / "coupling.npz").tocsr()
    assert matrix.shape == (15598, 15598)
    readouts = matrix[CORE:, :CORE]
    coords, dominant, radius = cloud_positions(matrix, nodes[:CORE], nodes[CORE:])
    fig, ax = plt.subplots(figsize=(180 / 25.4, 151 / 25.4))
    fig.subplots_adjust(left=.03, right=.97, top=.90, bottom=.085)
    ax.set_aspect("equal")
    bounds_low = (coords[:CORE] - radius[:, None]).min(axis=0)
    bounds_high = (coords[:CORE] + radius[:, None]).max(axis=0)
    ax.set(xlim=(bounds_low[0] - .14, bounds_high[0] + .14),
           ylim=(bounds_low[1] - .10, bounds_high[1] + .20))
    ax.set_axis_off()
    draw_edges(ax, matrix, coords, dominant)
    degree = readouts.getnnz(axis=1)
    ax.scatter(*coords[CORE:].T, s=.25 + .13 * degree, color="#294c78",
               alpha=.42, linewidths=0, zorder=3)
    ax.scatter(*coords[:CORE].T, s=18, facecolor="#f4f7fb", edgecolor="#274467",
               linewidths=.68, zorder=4)
    for i, name in enumerate(nodes[:CORE]):
        x, y = coords[i]
        ax.text(x, y + .77 * radius[i] + .054, name, ha="center", va="bottom", fontsize=7.3,
                weight="bold" if name in {"PRDM14", "TRIM25"} else "normal",
                color=BLUE if name == "PRDM14" else INK,
                bbox=dict(fc="white", ec="none", boxstyle="square,pad=.15", alpha=.93), zorder=8)
    draw_focus(ax, nodes, matrix, coords)

    fig.text(.048, .963, "Transcriptome-wide iDOP network", fontsize=10, weight="bold", va="top")
    fig.text(.048, .930, "Gravity-constrained layout  ·  23 groups  ·  15,598 genes  ·  115,369 edges",
             fontsize=8, color=MUTED, va="top")
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#f4f7fb",
               markeredgecolor="#274467", markeredgewidth=.7, markersize=4, label="Core gene"),
        Line2D([], [], marker="o", linestyle="none", color="#294c78", markersize=3, label="Readout gene"),
        Line2D([], [], color=BLUE, lw=1.05, label="Positive coupling"),
        Line2D([], [], color=ORANGE, lw=1.05, label="Negative coupling"),
    ]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(.5, .022),
               frameon=False, columnspacing=1.7, handlelength=1.7, fontsize=7.5)
    stem = OUT / "Figure_S4_idop_nebula"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()

