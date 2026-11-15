"""Screen all consistently detected RNA genes downstream of the CRISPR target network."""
import gzip
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.sparse.linalg import expm_multiply
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from idop_diffusion import TimeSeriesBatch, select_topology
from idop_diffusion.propagation import PropagationConfig, fit_propagation

RUN = ROOT / "results/idop_feasibility_20261109"
OUT = RUN / "transcriptome_readout"
DATA = ROOT / "data/GSE213069"
REAL = RUN / "real"
EXT = ROOT / "data/GSE303158"
DAYS = np.arange(2., 6.)
RIDGE = .1
BOOTSTRAPS = 32


def load_states(core, targets, readout_path=OUT / "eligible_readouts.csv", *, on_day=None):
    labels = [f"{target}_{arm}" for arm in (1, 2) for target in targets]
    groups_by_day, controls_by_day = [], []
    for day in DAYS.astype(int):
        path = DATA / "unpacked" / f"day{day}"
        meta = pd.read_csv(DATA / "processed" / f"cell_metadata_day{day}.csv.gz")
        barcodes = pd.read_csv(path / "barcodes.tsv.gz", header=None)[0]
        np.testing.assert_array_equal(meta.barcode, barcodes)
        features = pd.read_csv(path / "features.tsv.gz", sep="\t", header=None)
        rna = features[2].eq("Gene Expression")
        names, gene_index = np.unique(features.loc[rna, 1].to_numpy(str), return_inverse=True)
        if day == 2:
            genes = names
        np.testing.assert_array_equal(names, genes)

        qc = meta.pass_strict_qc.to_numpy(bool)
        group = pd.Index(labels).get_indexer(meta.top_guide)
        group[meta.target.eq("CONTROL")] = len(labels)
        columns = np.flatnonzero(qc & (group >= 0))
        sizes = np.bincount(group[columns], minlength=len(labels) + 1)
        assert (sizes >= 10).all()
        with gzip.open(path / "matrix.mtx.gz", "rb") as handle:
            matrix = mmread(handle, spmatrix=True).tocsr()[rna.to_numpy()][:, columns]
        merge = sparse.csr_matrix((np.ones(len(gene_index)), (gene_index, np.arange(len(gene_index)))),
                                  shape=(len(genes), len(gene_index)))
        matrix = (merge @ matrix).tocsc()
        matrix.eliminate_zeros()
        totals = meta.rna_umi.to_numpy()[columns]
        np.testing.assert_allclose(np.asarray(matrix.sum(axis=0)).ravel(), totals)
        controls_by_day.append(np.asarray(matrix[:, group[columns] == len(labels)].sum(axis=1)).ravel())
        matrix.data = np.log1p(matrix.data * np.repeat(1e4 / totals, np.diff(matrix.indptr)))
        if on_day is not None:
            on_day(day, genes, matrix, group[columns], labels, meta.barcode.to_numpy()[columns])
        aggregate = sparse.csr_matrix((1 / sizes[group[columns]], (np.arange(len(columns)), group[columns])),
                                      shape=(len(columns), len(sizes)))
        means = (matrix @ aggregate).toarray()
        groups_by_day.append((means[:, :-1] - means[:, -1:]).T)
        print(f"Day {day}: {len(columns)} cells", flush=True)

    controls = np.array(controls_by_day)
    measurable = (controls > 0).all(axis=0)
    core_idx = np.searchsorted(genes, core)
    np.testing.assert_array_equal(genes[core_idx], core)
    out_idx = np.flatnonzero(measurable & ~np.isin(genes, core))
    indices = np.r_[core_idx, out_idx]
    states = np.stack(groups_by_day, axis=1)[:, :, indices]
    for arm, reference in enumerate((1, 2)):
        ref = TimeSeriesBatch.load(REAL / f"reference_guide{reference}" / "training.npz")
        present = [gene for gene in ref.nodes if gene in set(genes[indices])]
        old = [ref.nodes.index(gene) for gene in present]
        new = [i for i in pd.Index(genes[indices]).get_indexer(present)]
        np.testing.assert_allclose(states[arm*len(targets):(arm+1)*len(targets), :, new],
                                   ref.states[:, :, old], atol=5e-7, rtol=0)
    gene_table = pd.DataFrame({"gene": genes[out_idx], **{
        f"control_UMI_day{int(day)}": controls[i, out_idx] for i, day in enumerate(DAYS)}})
    gene_table.to_csv(readout_path, index=False)
    return states, genes[out_idx], int(measurable.sum())


def standardize(x):
    scale = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(scale > 0, scale, 1.)


def output_support(states, nc):
    votes = np.zeros((states.shape[-1] - nc, nc), dtype=np.uint8)
    for i in range(len(DAYS) - 1):
        values = states[:, i:i+2].transpose(1, 0, 2).reshape(-1, states.shape[-1])
        z = standardize(values)
        design = np.asfortranarray(z[:, :nc])
        model = Lasso(alpha=.1, fit_intercept=False, max_iter=100_000, tol=1e-5)
        model.fit(design, z[:, nc:])
        votes += model.coef_ != 0
    return votes >= 2


def fit_response(states, core, source_idx, query_idx, *, topology=None, support=None):
    nc = len(core)
    batch = TimeSeriesBatch(states[:, :, :nc], DAYS, core, source_idx)
    if topology is None:
        topology = select_topology(batch)
    model = fit_propagation(batch, topology, PropagationConfig(stability_margin=None))
    if support is None:
        support = output_support(states, nc)
    scale = np.maximum(np.sqrt(np.mean(states**2, axis=(0, 1))), .02)
    z = states / scale
    x = (.5 * (z[:, 1:] + z[:, :-1])).reshape(-1, z.shape[-1])
    y = np.diff(z, axis=1).reshape(x.shape)
    xc = x[:, :nc]
    xo = x[:, nc:]
    yo = y[:, nc:]
    coef = np.empty((len(support), nc + 1))
    core_cov = xc.T @ xc / len(x)
    core_output = xc.T @ xo / len(x)
    core_rhs = xc.T @ yo / len(x)
    output_var = np.mean(xo**2, axis=0)
    output_rhs = np.mean(xo * yo, axis=0)
    for start in range(0, len(support), 512):
        span = slice(start, start + 512)
        mask = support[span]
        n = len(mask)
        system = np.zeros((n, nc + 1, nc + 1))
        system[:, :nc, :nc] = core_cov * mask[:, :, None] * mask[:, None, :]
        system[:, :nc, nc] = system[:, nc, :nc] = core_output[:, span].T * mask
        system[:, nc, nc] = output_var[span]
        system += RIDGE * np.eye(nc + 1)
        rhs = np.column_stack([core_rhs[:, span].T * mask, output_rhs[span]])
        coef[span] = np.linalg.solve(system, rhs[..., None])[..., 0]

    output_core = sparse.csr_matrix(coef[:, :nc] * scale[nc:, None] / scale[None, :nc])
    diagonal = coef[:, -1]
    shift = max(0., .01 + max(model.metadata["spectral_abscissa_before_projection"], diagonal.max()))
    coupling = sparse.bmat([
        [sparse.csr_matrix(model.matrix - shift * np.eye(nc)), None],
        [output_core, sparse.diags(diagonal - shift)],
    ], format="csr")
    inputs = sparse.csr_matrix((-np.ones(len(query_idx)), (query_idx, np.arange(len(query_idx)))),
                               shape=(coupling.shape[0], len(query_idx)))
    augmented = sparse.bmat([[coupling, inputs], [None, sparse.csr_matrix((len(query_idx), len(query_idx)))]],
                            format="csr")
    initial = np.vstack([np.zeros(inputs.shape), np.eye(len(query_idx))])
    response = expm_multiply(3 * augmented, initial)[nc:coupling.shape[0]].T
    return response, int(topology.mask.sum()), int(support.sum()), coupling


def nominate():
    protocol = json.loads((REAL / "protocol.json").read_text(encoding="utf-8"))
    core, targets = protocol["all_targets"], protocol["targets"]
    guides = json.loads((RUN / "joint_biology/frozen_external_guides.json").read_text())
    sources = sorted(guides)
    rules = dict(
        core=core, targets=targets, external_guides=guides, days=DAYS.tolist(),
        readouts="All non-core gene symbols with at least one RNA UMI in pooled strict-QC controls on each of days 2–5",
        predictors="Core-to-core and core-to-readout directed edges, with self terms in dynamics",
        normalization="Mean single-cell log1p CP10k minus pooled same-day non-targeting control mean",
        lasso_alpha=.1, window_days=2, minimum_windows=2, ridge=RIDGE, stability_margin=.01,
        forcing=-1, duration_days=3, bootstraps=BOOTSTRAPS, bootstrap_seed=20261109,
        minimum_abs_response=.03, minimum_sign_agreement=.8, maximum_nominees_per_source=5,
        external="Same directional CRISPR screen as fixed-panel run. Markers absent from the public RNA matrix are recorded as unavailable.",
    )
    (OUT / "protocol.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
    states, readouts, measurable = load_states(core, targets)
    indices = np.array([core.index(gene) for gene in targets] * 2)
    query = [core.index(gene) for gene in sources]
    response, n_core, n_readout, coupling = fit_response(states, core, indices, query)
    sparse.save_npz(OUT / "coupling.npz", coupling)
    np.save(OUT / "nodes.npy", np.r_[core, readouts])
    print(f"Readouts {len(readouts)}, edges {n_core}+{n_readout}. Bootstrap started.", flush=True)
    n = len(targets)
    matches = np.zeros(response.shape, dtype=np.uint8)
    rng = np.random.default_rng(20261109)
    for i in range(BOOTSTRAPS):
        ids = rng.choice(n, n, replace=True)
        keep = np.r_[ids, ids + n]
        boot, _, _, _ = fit_response(states[keep], core, indices[keep], query)
        matches += np.sign(boot) == np.sign(response)
        if (i + 1) % 4 == 0:
            print(f"Bootstrap {i + 1}/{BOOTSTRAPS}", flush=True)
    tables = []
    for i, source in enumerate(sources):
        table = pd.DataFrame(dict(source=source, marker=readouts, day5_response=response[i],
                                  sign_agreement=matches[i] / BOOTSTRAPS))
        table["abs_response"] = table.day5_response.abs()
        table = table.sort_values(["abs_response", "marker"], ascending=[False, True])
        table["rank"] = np.arange(1, len(table) + 1)
        table["eligible"] = (table.abs_response >= .03) & (table.sign_agreement >= .8)
        table["nominated"] = table.eligible & (table.eligible.cumsum() <= 5)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(OUT / "all_scored_pairs.csv", index=False)
    frozen = OUT / "frozen_candidates.csv"
    table.loc[table.nominated].to_csv(frozen, index=False)
    (OUT / "freeze.json").write_text(json.dumps(dict(
        measurable=measurable, readouts=len(readouts), pairs=len(table), core_edges=n_core,
        core_to_readout_edges=n_readout, nominees=int(table.nominated.sum()),
        frozen_candidates_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest()), indent=2),
        encoding="utf-8")
    print(table.loc[table.nominated | (table.source.eq("PRDM14") & table.marker.eq("DNMT3B"))]
          .to_string(index=False), flush=True)



def external_check():
    frozen = OUT / "frozen_candidates.csv"
    receipt = json.loads((OUT / "freeze.json").read_text())
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == receipt["frozen_candidates_sha256"]
    pairs = pd.read_csv(frozen)
    rank = pd.read_csv(OUT / "all_scored_pairs.csv")
    focal = rank.loc[rank.source.eq("PRDM14") & rank.marker.eq("DNMT3B") & ~rank.nominated]
    pairs = pd.concat([pairs, focal], ignore_index=True)
    guides = json.loads((OUT / "protocol.json").read_text())["external_guides"]
    result = check_external_pairs(pairs, guides)
    result.to_csv(OUT / "external_screen.csv", index=False)
    print(result[["source", "marker", "rank", "nominated", "external_status"]].to_string(index=False),
          flush=True)


def check_external_pairs(pairs, guides, *, unpacked_path=None):
    from inspect_gse303158_primed import (
        logmap_axis, named_pairs, rdata, slots, sparse_and_axis, vector,
    )
    meta = pd.read_csv(EXT / "primed_panel/cell_metadata.csv.gz").set_index("barcode")
    assert meta.num_features.eq(1).all() and meta.index.is_unique
    source_dir = EXT / "processed_source"
    files = json.loads((source_dir / "manifest.json").read_text())["files"]
    filename, provenance = next(iter(files.items()))
    archive = source_dir / filename
    assert archive.stat().st_size == provenance["bytes"]
    temp = ROOT / "runtime_temp/GSE303158_primed.RData"
    if not temp.exists():
        partial = temp.with_suffix(".part")
        with gzip.open(archive, "rb") as src, partial.open("wb") as dest:
            shutil.copyfileobj(src, dest, 4 * 1024**2)
        partial.replace(temp)

    if unpacked_path is not None:
        with temp.open("rb") as handle:
            compressed = handle.read(2) == b"\x1f\x8b"
        if compressed:
            partial = unpacked_path.with_suffix(".part")
            with gzip.open(temp, "rb") as src, partial.open("wb") as dest:
                shutil.copyfileobj(src, dest, 4 * 1024**2)
            partial.replace(unpacked_path)
            temp = unpacked_path

    print("Reading GSE303158 RNA for frozen pairs", flush=True)
    with temp.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        objects = named_pairs(rdata.parser.parse_data(data).object)
        seurat, = [slots(obj) for obj in objects.values()
                   if {"meta.data", "assays"} <= set(slots(obj))]
        rna = slots(vector(seurat["assays"])["RNA"])
        if "counts" in rna:
            matrices = [sparse_and_axis(rna["counts"])]
        else:
            matrices = [
                sparse_and_axis(node, logmap_axis(rna["features"], layer),
                                logmap_axis(rna["cells"], layer))
                for layer, node in vector(rna["layers"]).items()
                if layer == "counts" or layer.startswith("counts.")
            ]
        genes = sorted(set(pairs.marker).intersection(*(set(features) for _, features, _ in matrices)))
        values = []
        for matrix, features, cells in matrices:
            gene_indices = pd.Index(features).get_indexer(genes)
            cell_meta = meta.loc[cells]
            totals = np.asarray(matrix.sum(axis=0)).ravel()
            np.testing.assert_array_equal(totals, cell_meta.nCount_RNA)
            values.append(pd.DataFrame(np.log1p(matrix[gene_indices].toarray().T * (1e4 / totals[:, None])),
                                       columns=genes, index=cells))
        values = pd.concat(values).loc[meta.index]
        del matrices, rna, seurat, objects

    records = []
    for row in pairs.itertuples(index=False):
        result = row._asdict()
        if row.marker not in values:
            records.append({**result, "external_status": "unavailable"})
            continue
        signal = values[row.marker]
        differences, statistics = [], []
        for guide in guides[row.source]:
            for batch in sorted(meta.batch.unique()):
                same_batch = meta.batch.eq(batch)
                treated = signal[meta.feature_call.eq(guide) & same_batch]
                control = signal[meta.target.eq("CTRL") & same_batch]
                assert len(treated) >= 10 and len(control) >= 20
                delta = treated.mean() - control.mean()
                se = np.sqrt(max(treated.var(ddof=1) / len(treated) +
                                 control.var(ddof=1) / len(control), 1e-12))
                differences.append(delta)
                statistics.append(delta / se)
        direction = np.sign(row.day5_response)
        raw = np.array(differences).reshape(2, 2)
        directed = direction * raw
        min_stat = (direction * np.array(statistics).reshape(2, 2).mean(axis=1)).min()
        passes = (directed > 0).all() and (directed.mean(axis=1) >= .10).all() and min_stat >= 2
        records.append({**result, "external_status": "pass" if passes else "fail",
                        "external_min_delta": raw.min(), "external_max_delta": raw.max(),
                        "external_mean_delta": raw.mean(), "external_min_directed_guide_t": min_stat,
                        "guide_batch_deltas": json.dumps(differences)})
    return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        if not args.external_only:
            nominate()
        external_check()

