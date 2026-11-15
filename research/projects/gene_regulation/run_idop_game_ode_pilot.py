"""Fit and audit real-time iDOP-style additive dynamics on the strict CRISPR cohort."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.integrate import simpson, solve_ivp
from threadpoolctl import threadpool_limits

from screen_idop_transcriptome_readouts import (
    DAYS, REAL, RUN, TimeSeriesBatch, load_states, output_support, select_topology,
)


RIDGE = .1
CURVATURE = (.1, 1., 10., np.inf)
DOSE = .5  # training expression scale / day
BOOTSTRAPS = 200
OUT = RUN / "game_ode_pilot_20261110"


@dataclass
class Dynamics:
    times: np.ndarray
    control: np.ndarray
    scale: np.ndarray
    linear: np.ndarray
    quadratic: np.ndarray
    bounds: np.ndarray
    forcing: np.ndarray
    condition: np.ndarray
    rank: np.ndarray

    def control_at(self, t):
        idx = np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times) - 2)
        fraction = (t - self.times[idx]) / (self.times[idx + 1] - self.times[idx])
        return self.control[idx] + fraction * (self.control[idx + 1] - self.control[idx])

    def quadratic_basis(self, z):
        # C1 tangent continuation outside the fitted range. No quadratic runaway.
        boundary = np.clip(z, *self.bounds)
        return boundary**2 + 2 * boundary * (z - boundary)

    def slope(self, z):
        return self.linear + 2 * self.quadratic * np.clip(z, *self.bounds)

    def forecast(self, initial, forcing, *, local=False, receivers=()):
        def rhs(t, delta):
            control = self.control_at(t)
            if local:
                return self.slope(control) @ delta + forcing
            curvature = self.quadratic_basis(control + delta) - self.quadratic_basis(control)
            return self.linear @ delta + self.quadratic @ curvature + forcing

        sol = solve_ivp(rhs, self.times[[0, -1]], initial, t_eval=self.times,
                        rtol=2e-7, atol=1e-9, max_step=.2, dense_output=True)
        if not sol.success or not np.isfinite(sol.y).all():
            raise RuntimeError(sol.message)
        grid = np.linspace(self.times[0], self.times[-1], 61)
        delta = sol.sol(grid).T
        control = np.array([self.control_at(t) for t in grid])
        states = delta + control
        outside = (states < self.bounds[0] - 1e-8) | (states > self.bounds[1] + 1e-8)
        if len(receivers):
            if local:
                raise ValueError("Attribution is defined for the full dynamics")
            effects = self.linear[receivers] * simpson(delta, x=grid, axis=0)
            effects += self.quadratic[receivers] * simpson(
                self.quadratic_basis(states) - self.quadratic_basis(control), x=grid, axis=0)
            np.testing.assert_allclose(effects.sum(axis=1) + forcing[receivers] * np.ptp(self.times),
                                       sol.y[receivers, -1] - initial[receivers], atol=2e-6, rtol=3e-4)
            return sol.y.T, outside, effects * self.scale[receivers, None]
        return sol.y.T, outside


def integrated_design(delta, control, times, bounds=None):
    dt = np.diff(times)[None, :, None]
    left, right = delta[:, :-1], delta[:, 1:]
    cl, cr = control[:-1], control[1:]
    # Simpson integrates the squared difference exactly on linear interpolants.
    ql, qr = left * (2 * cl + left), right * (2 * cr + right)
    middle, cm = (left + right) / 2, (cl + cr) / 2
    qm = middle * (2 * cm + middle)
    quadratic = (ql + 4 * qm + qr) / 6
    if bounds is not None:
        start, end = cl + left, cr + right

        def primitive(z):
            boundary = np.clip(z, *bounds)
            d = z - boundary
            return boundary**3 / 3 + boundary**2 * d + boundary * d**2

        step = end - start
        steady = abs(step) < 1e-8
        mid = (start + end) / 2
        boundary = np.clip(mid, *bounds)
        mean_basis = np.divide(primitive(end) - primitive(start), step,
                               out=boundary**2 + 2 * boundary * (mid - boundary), where=~steady)
        quadratic = mean_basis - (cl**2 + cl * cr + cr**2) / 3
    p = delta.shape[-1]
    return ((left + right) * dt / 2).reshape(-1, p), (quadratic * dt).reshape(-1, p)


def fit(states, control, times, sources, support, penalty, core_size):
    p = states.shape[-1]
    scale = np.maximum(np.sqrt(np.mean(states**2, axis=(0, 1))), .02)
    delta = states / scale
    control = (control - control.mean(axis=0)) / scale
    absolute = np.concatenate([control[None], control + delta], axis=0)
    bounds = np.array([absolute.min(axis=(0, 1)), absolute.max(axis=(0, 1))])
    x, q = integrated_design(delta, control, times)
    y = np.diff(delta, axis=1).reshape(-1, p)
    src = np.repeat(sources, len(times) - 1)
    dt = np.tile(np.diff(times), len(sources))
    linear, quadratic = np.zeros((p, p)), np.zeros((p, p))
    forcing, condition, rank = np.zeros(p), np.empty(p), np.empty(p, int)
    for i in range(p):
        incoming = np.flatnonzero(support[i] | (np.arange(p) == i))
        use_q = np.isfinite(penalty) and i < core_size
        design = x[:, incoming]
        penalties = [RIDGE] * len(incoming)
        if use_q:
            design = np.column_stack([design, q[:, incoming]])
            penalties.extend([penalty] * len(incoming))
        active = np.any(src == i)
        if active:
            design = np.column_stack([design, dt * (src == i)])
            penalties.append(RIDGE)
        singular = np.linalg.svd(design, compute_uv=False)
        rank[i] = np.sum(singular > singular[0] * max(design.shape) * np.finfo(float).eps)
        condition[i] = singular[0] / singular[-1] if singular[-1] > 0 and rank[i] == design.shape[1] else np.inf
        gram = design.T @ design / len(y) + np.diag(penalties)
        rhs = design.T @ y[:, i] / len(y)
        coef = np.linalg.solve(gram, rhs)
        if active and coef[-1] > 0:
            # Constrain the direct input to nonpositive values. Refit at the boundary.
            coef[:-1] = np.linalg.solve(gram[:-1, :-1], rhs[:-1])
            coef[-1] = 0.
        if active:
            forcing[i] = coef[-1]
        linear[i, incoming] = coef[:len(incoming)]
        if use_q:
            quadratic[i, incoming] = coef[len(incoming):2 * len(incoming)]
    return Dynamics(times, control, scale, linear, quadratic, bounds, forcing, condition, rank)


def select(states, sources, genes, core_size):
    batch = TimeSeriesBatch(states[:, :, :core_size], DAYS, genes[:core_size], sources)
    support = np.zeros((len(genes), len(genes)), dtype=bool)
    support[:core_size, :core_size] = select_topology(batch).mask
    if len(genes) > core_size:
        support[core_size:, :core_size] = output_support(states, core_size)
    return support


def tune(states, control, sources, genes):
    errors = []
    for hold in np.unique(sources):
        train = sources != hold
        support = select(states[train], sources[train], genes, len(genes))
        row = []
        for penalty in CURVATURE:
            model = fit(states[train], control, DAYS, sources[train], support, penalty, len(genes))
            truth = states[~train] / model.scale
            x, q = integrated_design(truth, model.control, DAYS, model.bounds)
            residual = (x @ model.linear.T + q @ model.quadratic.T
                        - np.diff(truth, axis=1).reshape(-1, len(genes)))
            row.append(np.mean(residual[:, np.arange(len(genes)) != hold]**2))
        errors.append(row)
    errors = np.array(errors)
    best = errors.mean(axis=0).argmin()
    # Prefer the strongest shrinkage within one paired SE of the best candidate.
    delta = errors - errors[:, [best]]
    acceptable = delta.mean(axis=0) <= delta.std(axis=0, ddof=1) / np.sqrt(len(errors))
    return CURVATURE[np.flatnonzero(acceptable)[-1]], errors.mean(axis=0).tolist()


def load_panel(case_path, core, targets, scratch):
    panel = json.loads(case_path.read_text(encoding="utf-8"))
    genes = np.array(panel["genes"])
    nc = len(core)
    np.testing.assert_array_equal(genes[:nc], core)
    controls = []

    def collect(day, names, matrix, groups, labels, barcodes):
        indices = np.searchsorted(names, genes)
        np.testing.assert_array_equal(names[indices], genes)
        controls.append(np.asarray(matrix[indices][:, groups == len(labels)].mean(axis=1)).ravel())

    all_states, readouts, _ = load_states(core, targets, scratch / "eligible_readouts.csv", on_day=collect)
    columns = np.r_[np.arange(nc), nc + np.searchsorted(readouts, genes[nc:])]
    np.testing.assert_array_equal(np.r_[core, readouts][columns], genes)
    return all_states[:, :, columns], np.array(controls), genes, panel["expected"]


def simulation():
    # Exact nonuniform-time basis check for delta=t and control=t, both on [0, 2].
    t = np.array([0., .25, 2.])
    x, q = integrated_design(t[None, :, None], t[:, None], t)
    np.testing.assert_allclose(x[:, 0], np.diff(t**2 / 2))
    np.testing.assert_allclose(q[:, 0], np.diff(t**3))
    _, continued = integrated_design(t[None, :, None], t[:, None], t, np.array([[-1.], [2.]]))
    np.testing.assert_allclose(continued[:, 0], [1 / 64, 8 - 1 / 64 - 4 / 3])
    rng = np.random.default_rng(6605)
    times = DAYS.copy()
    controls = .12 * (times[:, None] - times.mean()) * np.array([1., -.7, .4])
    a = np.array([[-.3, 0., 0.], [.8, -.3, 0.], [0., -.65, -.4]])
    support = (a != 0) & ~np.eye(3, dtype=bool)
    sources = np.tile([-1, 0, 1, 2], 8)
    results = []
    for curvature in (0., .45):
        b = np.zeros_like(a)
        b[1, 0] = curvature
        true = Dynamics(times, controls, np.ones(3), a, b, np.array([[-4.] * 3, [4.] * 3]),
                        np.full(3, -.25), np.ones(3), np.ones(3))
        initial = rng.uniform(-.9, .9, (len(sources), 3))
        u = -.25 * (sources[:, None] == np.arange(3))
        trajectories = np.array([true.forecast(x0, force)[0] for x0, force in zip(initial, u)])
        query = np.array([-.25, 0., 0.])
        expected = true.forecast(np.zeros(3), query)[0][-1]
        for noise in (0., .025):
            noisy = trajectories + rng.normal(0., noise, trajectories.shape)
            for penalty in (np.inf, .1):
                model = fit(noisy, controls, times, sources, support, penalty, 3)
                prediction = model.forecast(np.zeros(3), query / model.scale)[0][-1] * model.scale
                np.testing.assert_allclose(model.forecast(np.zeros(3), np.zeros(3))[0], 0.)
                true_direction = np.sign(expected[1:])
                assert (np.sign(prediction[1:]) == true_direction).all()
                results.append(dict(curvature=curvature, noise=noise, penalty=str(penalty),
                                    slope_0_to_1=float(model.linear[1, 0] * model.scale[1] / model.scale[0]),
                                    curvature_0_to_1=float(model.quadratic[1, 0] * model.scale[1] / model.scale[0]**2),
                                    endpoint_error=float(np.linalg.norm(expected - prediction))))
    return results


def evaluate(states, control, genes, sources, expected, simulation_results):
    nc = 23
    core, core_states, core_control = genes[:nc], states[:, :, :nc], control[:, :nc]
    forecasts, cases, folds, contributions = [], [], [], []
    for hold in np.unique(sources):
        train = sources != hold
        penalty, inner_scores = tune(core_states[train], core_control, sources[train], core)
        support = select(core_states[train], sources[train], core, nc)
        fold = dict(source=core[hold], penalty=str(penalty), inner_increment_loss=inner_scores,
                    edges=int(support.sum()))
        folds.append(fold)
        idx = np.arange(nc) != hold
        relevant = [case for case in expected if case["source"] == core[hold]]
        panel_support = select(states[train], sources[train], genes, nc) if relevant else None
        for label, value in (("linear", np.inf), ("quadratic", penalty), ("quadratic_fixed", .1)):
            model = fit(core_states[train], core_control, DAYS, sources[train], support, value, nc)
            target_results = []
            for measured in core_states[~train]:
                initial = measured[0] / model.scale
                # No fitted source forcing for a wholly held-out target.
                forecast, outside = model.forecast(initial, np.zeros(nc))
                truth = (measured[1:] / model.scale)[..., idx]
                prediction = forecast[1:, idx]
                error = np.mean((truth - prediction)**2)
                denom = np.linalg.norm(truth) * np.linalg.norm(prediction)
                target_results.append((error, float(np.sum(truth * prediction) / denom), outside[:, idx].mean()))
            error, cosine, outside = np.mean(target_results, axis=0)
            forecasts.append(dict(source=core[hold], model=label, mse=error, cosine=cosine,
                                  fraction_outside=outside, max_condition=float(model.condition.max())))
            if relevant:
                panel_model = fit(states[train], control, DAYS, sources[train], panel_support, value, nc)
                force = -DOSE * (np.arange(len(genes)) == hold)
                modes = (False, True) if label == "quadratic" else (False,)
                for local in modes:
                    receivers = np.array([np.flatnonzero(genes == case["marker"])[0] for case in relevant])
                    if label == "quadratic" and not local:
                        virtual, out_of_range, effects = panel_model.forecast(
                            np.zeros(len(genes)), force, receivers=receivers)
                        for receiver, row in zip(receivers, effects):
                            for sender in np.flatnonzero(row):
                                contributions.append(dict(source=genes[hold], marker=genes[receiver],
                                                          sender=genes[sender], effect=row[sender],
                                                          term="self" if sender == receiver else "core"))
                    else:
                        virtual, out_of_range = panel_model.forecast(np.zeros(len(genes)), force, local=local)
                    response = virtual[-1, nc:] * panel_model.scale[nc:]
                    ranks = np.argsort(np.argsort(-np.abs(response), kind="stable")) + 1
                    for case in relevant:
                        j = int(np.flatnonzero(genes[nc:] == case["marker"])[0])
                        direction = int(np.sign(response[j]))
                        cases.append(dict(**case, model="local_jacobian" if local else label,
                                          response=response[j], rank=int(ranks[j]),
                                          direction=direction, signed_top5=bool(ranks[j] <= 5 and direction == case["expected_direction"]),
                                          core_fraction_outside=float(out_of_range[:, :nc].mean())))
        print(f"Hold {core[hold]}: curvature penalty {penalty}, {support.sum()} edges", flush=True)

    penalty, full_scores = tune(core_states, core_control, sources, core)
    support = select(core_states, sources, core, nc)
    full = fit(core_states, core_control, DAYS, sources, support, penalty, nc)
    rng = np.random.default_rng(20261110)
    unique = np.unique(sources)
    target_rows = {i: np.flatnonzero(sources == i) for i in unique}
    weights, selected = [], []
    for draw in rng.choice(unique, (BOOTSTRAPS, len(unique)), replace=True):
        rows = np.concatenate([target_rows[i] for i in draw])
        mask = select(core_states[rows], sources[rows], core, nc)
        model = fit(core_states[rows], core_control, DAYS, sources[rows], mask, penalty, nc)
        weights.append(model.slope(model.control.mean(axis=0)) * model.scale[:, None] / model.scale)
        selected.append(mask)
    weights, selected = np.array(weights), np.array(selected)
    weight = full.slope(full.control.mean(axis=0)) * full.scale[:, None] / full.scale
    edges = []
    for i, j in np.argwhere(support):
        lo, hi = np.quantile(weights[:, i, j], [.025, .975])
        agreement = np.mean(selected[:, i, j] & (np.sign(weights[:, i, j]) == np.sign(weight[i, j])))
        edges.append(dict(source=core[j], target=core[i], weight=weight[i, j],
                          bootstrap_q025=lo, bootstrap_q975=hi,
                          bootstrap_selected=float(selected[:, i, j].mean()),
                          bootstrap_signed=float(agreement), stable=bool(agreement >= .8),
                          curvature=full.quadratic[i, j] * full.scale[i] / full.scale[j]**2))
    frame = pd.DataFrame(forecasts)
    pivot = frame.pivot(index="source", columns="model", values="mse")
    paired = (pivot.quadratic - pivot.linear).to_numpy()
    interval = np.quantile(rng.choice(paired, (10000, len(paired)), replace=True).mean(1), [.025, .975])
    summary = dict(simulation=simulation_results, folds=folds, full_penalty=str(penalty),
                   full_inner_increment_loss=full_scores, edges=len(edges), stable_edges=sum(e["stable"] for e in edges),
                   condition_median=float(np.median(full.condition)), condition_max=float(full.condition.max()),
                   forcing_at_zero=int(np.sum(full.forcing[unique] == 0)),
                   max_real_eigenvalue=float(np.linalg.eigvals(full.slope(full.control.mean(axis=0))).real.max()),
                   averages=frame.groupby("model")[["mse", "cosine", "fraction_outside"]].mean().to_dict("index"),
                   quadratic_wins_ties_losses=[int((paired < -1e-10).sum()), int((abs(paired) <= 1e-10).sum()), int((paired > 1e-10).sum())],
                   quadratic_minus_linear_mse_interval=interval.tolist(), dose=DOSE, bootstraps=BOOTSTRAPS)
    return summary, pd.DataFrame(edges), pd.DataFrame(cases), pd.DataFrame(contributions), full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--save", action="store_true", help="Save compact audit and weights after reviewing the pilot")
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    protocol = json.loads((REAL / "protocol.json").read_text(encoding="utf-8"))
    core, targets = protocol["all_targets"], protocol["targets"]
    case_path = RUN / "renge_108_case_followup_20261110/case_protocol.json"
    sources = np.tile([core.index(source) for source in targets], 2)
    with threadpool_limits(1):
        simulation_results = simulation()
        states, control, genes, expected = load_panel(case_path, core, targets, args.scratch)
        summary, edges, cases, contributions, full = evaluate(states, control, genes, sources, expected, simulation_results)
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2), flush=True)
    print(cases.to_string(index=False), flush=True)
    print(edges.nlargest(10, "bootstrap_signed").to_string(index=False), flush=True)
    print(f"Elapsed {perf_counter() - start:.1f} seconds", flush=True)
    if args.save:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        edges.to_csv(OUT / "core_edges.csv", index=False)
        cases.to_csv(OUT / "retrospective_cases.csv", index=False)
        contributions.to_csv(OUT / "retrospective_contributions.csv", index=False)
        np.savez_compressed(OUT / "core_model.npz", genes=genes[:len(core)], **vars(full))


if __name__ == "__main__":
    main()
