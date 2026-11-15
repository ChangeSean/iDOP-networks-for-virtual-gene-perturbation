"""Recreate the numerical results and figures in the iDOP review archive."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research/projects/gene_regulation"))
import screen_idop_transcriptome_readouts as analysis

RUN = ROOT / "results/idop_feasibility_20261109"
FIGURES = RUN / "manuscript_figures"
SOURCE = ROOT / "source_data"
PROTOCOL = json.loads((analysis.OUT / "protocol.json").read_text(encoding="utf-8"))
CORE = PROTOCOL["core"]
TARGETS = PROTOCOL["targets"]
NANOG = CORE.index("NANOG")
POU5F1 = CORE.index("POU5F1")


def nanog_contributions(coupling, duration, *, close_edge=False):
    n = len(CORE)
    assert coupling[:n, n:].nnz == 0
    matrix = coupling[:n, :n].toarray()
    if close_edge:
        matrix[POU5F1, NANOG] = 0.
    system = np.zeros((2 * n + 1, 2 * n + 1))
    system[:n, :n] = matrix
    system[NANOG, -1] = PROTOCOL["forcing"]
    system[n:2 * n, :n] = np.eye(n)
    state = expm(duration * system)[:, -1]
    terms = matrix[POU5F1] * state[n:2 * n]
    np.testing.assert_allclose(terms.sum(), state[POU5F1], atol=1.e-12)
    return state[POU5F1], terms


def stored_results():
    coupling = sparse.load_npz(analysis.OUT / "coupling.npz").tocsr()
    nodes = np.load(analysis.OUT / "nodes.npy", allow_pickle=False)
    full = pd.read_csv(analysis.OUT / "all_scored_pairs.csv", keep_default_na=False)
    screen = pd.read_csv(analysis.OUT / "external_screen.csv", keep_default_na=False)
    freeze = json.loads((analysis.OUT / "freeze.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((analysis.OUT / "frozen_candidates.csv").read_bytes()).hexdigest()
    assert digest == freeze["frozen_candidates_sha256"]
    np.testing.assert_array_equal(nodes[:len(CORE)], CORE)
    assert coupling.shape == (15598, 15598)
    assert coupling.nnz - np.count_nonzero(coupling.diagonal()) == 115369
    assert (len(full), int(full.eligible.sum()), int(full.nominated.sum())) == (62300, 1288, 20)
    assert int(screen.external_status.eq("pass").sum()) == 7

    expected = pd.read_csv(SOURCE / "NANOG_full_model.csv")
    observed = []
    for t in expected["simulation_day"]:
        total, terms = nanog_contributions(coupling, t)
        closed, _ = nanog_contributions(coupling, t, close_edge=True)
        observed.append((total, terms[NANOG], terms[POU5F1],
                         terms.sum() - terms[NANOG] - terms[POU5F1], closed))
    np.testing.assert_allclose(expected.iloc[:, 1:].to_numpy(), observed, atol=1.e-12)
    external = pd.read_csv(SOURCE / "NANOG_GSE303158_source.csv")
    assert len(external) == 8 and external.POU5F1_contrast.lt(0).all()
    np.testing.assert_allclose(external.POU5F1_mean - external.POU5F1_control,
                               external.POU5F1_contrast, atol=1.e-12)
    print("Verified 115,369 edges, 62,300 scores, 1,288 eligible pairs, 20 nominations, "
          "seven external screen passes and the NANOG figure-model calculation.", flush=True)
    return coupling, nodes, full


def refit_results(coupling, nodes, full):
    with np.load(SOURCE / "training_states.npz", allow_pickle=False) as data:
        states, names = data["states"], data["genes"]
    np.testing.assert_array_equal(names, nodes[len(CORE):])
    assert states.shape == (32, 4, 15598)
    ids = np.tile([CORE.index(g) for g in TARGETS], 2)
    queries = sorted(PROTOCOL["external_guides"])
    query_indices = [CORE.index(g) for g in queries]
    with threadpool_limits(1):
        responses, _, _, matrix = analysis.fit_response(states, CORE, ids, query_indices)
        np.testing.assert_allclose((matrix - coupling).data, 0., atol=1.e-12)
        expected = full.pivot(index="source", columns="marker", values="day5_response")
        np.testing.assert_allclose(responses, expected.loc[queries, names], atol=1.e-12)
        archived = pd.read_csv(FIGURES / "Figure_3_genomewide_heldout_scores.csv", keep_default_na=False)
        for source in ("PRDM14", "POU5F1"):
            i = CORE.index(source)
            keep = ids != i
            fit, _, _, _ = analysis.fit_response(states[keep], CORE, ids[keep], [i])
            ref = archived.loc[archived.source.eq(source)].set_index("readout").loc[names]
            np.testing.assert_allclose(fit[0], ref.heldout_response, atol=1.e-12)
        print("Full fit and both target holdouts regenerated.", flush=True)
        rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
        n = len(TARGETS)
        agreements = np.zeros_like(responses, dtype=int)
        summary = []
        for iteration in range(PROTOCOL["bootstraps"]):
            sampled = rng.choice(n, n, replace=True)
            rows = np.r_[sampled, sampled + n]
            boot, _, _, matrix = analysis.fit_response(states[rows], CORE, ids[rows], query_indices)
            agreements += np.sign(boot) == np.sign(responses)
            y, terms = nanog_contributions(matrix, PROTOCOL["duration_days"])
            summary.append((np.argmax(abs(terms)) == NANOG, y < 0))
            if (iteration + 1) % 8 == 0:
                print(f"Bootstrap refits {iteration+1}/{PROTOCOL['bootstraps']}", flush=True)
    expected = full.pivot(index="source", columns="marker", values="sign_agreement")
    np.testing.assert_allclose(agreements / PROTOCOL["bootstraps"], expected.loc[queries, names])
    np.testing.assert_array_equal(np.sum(summary, axis=0), (29, 30))
    print("Matched all bootstrap sign agreements and recovered NANOG counts 29/32 and 30/32.",
          flush=True)


def regenerate_figures(results):
    import matplotlib
    matplotlib.use("Agg")
    from plot_idop_aligned_figures import workflow
    from plot_idop_full_decomposition import main as fig2
    from plot_paired_perturbation_evidence import main as fig34
    import build_followup_figures_20261112 as followup

    readouts = pd.read_csv(analysis.OUT / "eligible_readouts.csv", keep_default_na=False)
    full = results[2]
    followup.OUT.mkdir(parents=True, exist_ok=True)
    workflow(PROTOCOL, readouts, full)
    fig2()
    fig34()
    followup.figure6(pd.read_csv(SOURCE / "Figure_5_Hallmark_screen.csv"))
    print("Regenerated all five main figures from the archived inputs.", flush=True)


def verify_hallmark():
    import build_followup_figures_20261112 as followup
    protocol = json.loads((SOURCE / "Figure_5_Hallmark_protocol.json").read_text(encoding="utf-8"))
    if not followup.GMT.is_file() or not followup.FENG.is_file():
        raise FileNotFoundError("Place the original Enrichr and Figshare files at the paths in DATA_SOURCES.md.")
    assert hashlib.sha256(followup.GMT.read_bytes()).hexdigest() == protocol["gmt_sha256"]
    followup.STAGE.mkdir(parents=True, exist_ok=True)
    reproduced = followup.hallmark_screen()
    expected = pd.read_csv(SOURCE / "Figure_5_Hallmark_screen.csv")
    pd.testing.assert_frame_equal(reproduced, expected, check_exact=False, rtol=1.e-10, atol=1.e-12)
    print("Recalculated the full Hallmark screen from the original Enrichr and Figshare inputs.",
          flush=True)


def verify_supplement():
    import run_idop_game_ode_pilot as supplement

    case_file = RUN / "renge_108_case_followup_20261110/case_protocol.json"
    case = json.loads(case_file.read_text(encoding="utf-8"))
    with np.load(SOURCE / "supplement_108_panel.npz", allow_pickle=False) as data:
        states, control, genes = (data[k] for k in ("states", "control", "genes"))
    np.testing.assert_array_equal(genes, case["genes"])
    ids = np.tile([CORE.index(g) for g in TARGETS], 2)
    with threadpool_limits(1):
        summary, _, evaluated, _, _ = supplement.evaluate(
            states, control, genes, ids, case["expected"], []
        )
    expected = pd.read_csv(supplement.OUT / "retrospective_cases.csv")
    pd.testing.assert_frame_equal(expected, evaluated, check_exact=False, rtol=1.e-9, atol=1.e-12)
    frozen = json.loads((supplement.OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["full_penalty"] == frozen["full_penalty"] == "inf"
    print("Regenerated the 108-gene sensitivity analysis and all five reported directions.",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true", help="Re-estimate the model from archived guide trajectories.")
    parser.add_argument("--figures", action="store_true", help="Regenerate figures using archived source data.")
    parser.add_argument("--hallmark", action="store_true", help="Recalculate Fig. 5 from downloaded input files.")
    parser.add_argument("--supplement", action="store_true", help="Recalculate the 108-gene sensitivity analysis.")
    args = parser.parse_args()
    results = stored_results()
    if args.refit:
        refit_results(*results)
    if args.figures:
        regenerate_figures(results)
    if args.hallmark:
        verify_hallmark()
    if args.supplement:
        verify_supplement()


if __name__ == "__main__":
    main()

