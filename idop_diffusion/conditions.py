"""Independent linear propagation with general condition and known-context inputs."""
from dataclasses import dataclass, field
import json
import numpy as np
import pandas as pd
from scipy.linalg import expm
from .topology import Topology, EdgeConfig
from ._vendor.idop_edge_select import edge_select


def observed_topology(states, times, nodes, config=EdgeConfig()):
    """Apply the original selector only to complete observed rows in actual-time windows."""
    width = config.window_points or len(times)
    masks, bounds = [], []
    for left in range(0, len(times) - width + 1, config.stride):
        rows = states[:, left:left + width].reshape(-1, len(nodes))
        rows = rows[np.isfinite(rows).all(axis=1)]
        if len(rows) < config.min_rows:
            raise ValueError('Too few complete observations in an actual-time window')
        selected = edge_select(pd.DataFrame(rows, columns=nodes), alpha=config.alpha, k=1, threshold=0.)
        mask = np.zeros((len(nodes), len(nodes)), dtype=bool)
        lookup = {name: index for index, name in enumerate(nodes)}
        for row in selected.itertuples(index=False):
            for source in row.source.strip('{}').split(','):
                if source:
                    mask[lookup[row.target], lookup[source]] = True
        masks.append(mask)
        bounds.append([times[left], times[left + width - 1]])
    frequency = np.mean(masks, axis=0)
    return Topology(tuple(nodes), frequency > config.threshold, frequency, np.array(masks), np.array(bounds),
        dict(selector='unchanged IDOP selector on complete observed rows', interpolation=False, numerical_weights_used=False))


@dataclass
class ConditionDynamics:
    nodes: tuple
    condition_names: tuple
    context_names: tuple
    null_condition: str
    reference_context: str
    support: np.ndarray
    matrix: np.ndarray
    condition_input: np.ndarray
    context_input: np.ndarray
    offset: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        p = len(self.nodes)
        if self.matrix.shape != (p, p) or self.support.shape != (p, p):
            raise ValueError('Incorrect state matrix shape')
        if self.condition_input.shape != (p, len(self.condition_names)) or self.context_input.shape != (p, len(self.context_names)):
            raise ValueError('Incorrect general input shape')
        if any(value.shape != (p,) for value in [self.offset, self.mean, self.scale]):
            raise ValueError('Incorrect preprocessing/input shape')
        arrays = [self.matrix, self.condition_input, self.context_input, self.offset, self.mean, self.scale]
        if any(not np.isfinite(value).all() for value in arrays) or np.any(self.scale <= 0):
            raise ValueError('Invalid model parameters')
        if np.any(np.abs(self.matrix[~(self.support | np.eye(p, dtype=bool))]) > 1e-12):
            raise ValueError('State dynamics violates binary support')
        if np.any(self.condition_input[:, self.condition_names.index(self.null_condition)] != 0):
            raise ValueError('Null condition must have zero direct input')

    def operators(self, duration):
        duration = float(duration)
        if not np.isfinite(duration) or duration < 0:
            raise ValueError('Invalid elapsed duration')
        if duration not in self._cache:
            p = len(self.nodes)
            block = np.zeros((2*p, 2*p))
            block[:p, :p] = self.matrix
            block[:p, p:] = np.eye(p)
            value = expm(block * duration)
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite propagation')
            self._cache[duration] = value[:p, :p], value[:p, p:]
        return self._cache[duration]

    def forcing(self, condition, context):
        return self.offset + self.condition_input[:, self.condition_names.index(condition)] + self.context_input[:, self.context_names.index(context)]

    def predict(self, initial, times, condition, context, initial_time=0.):
        initial = np.asarray(initial, dtype=float)
        if initial.shape != self.mean.shape or not np.isfinite(initial).all():
            raise ValueError('A complete observed initial state is required')
        standardized = (initial - self.mean) / self.scale
        force = self.forcing(condition, context)
        values = []
        for time in times:
            phi, integral = self.operators(time - initial_time)
            values.append((phi @ standardized + integral @ force) * self.scale + self.mean)
        return np.array(values)

    def trace_conditions(self, observed, times, context, initial_ridge=1.):
        """Receive no true condition; eliminate a common unknown anchor state for all candidates."""
        observed = np.asarray(observed, dtype=float)
        times = np.asarray(times, dtype=float)
        if observed.shape != (len(times), len(self.nodes)) or np.isinf(observed).any():
            raise ValueError('Invalid observed trajectory')
        if len(times) < 1 or np.any(np.diff(times) <= 0) or not np.isfinite(times).all() or initial_ridge <= 0:
            raise ValueError('Invalid times or initial-state penalty')
        operators = [self.operators(time - times[0]) for time in times]
        homogeneous = np.vstack([phi for phi, integral in operators])
        forcing = np.column_stack([self.forcing(condition, context) for condition in self.condition_names])
        signatures = np.vstack([integral @ forcing for phi, integral in operators])
        values = ((observed - self.mean) / self.scale).ravel()
        valid = np.isfinite(values)
        if valid.sum() < 2:
            raise ValueError('Insufficient observed values')
        homogeneous = homogeneous[valid]
        differences = values[valid, None] - signatures[valid]
        initial = np.linalg.solve(homogeneous.T @ homogeneous + initial_ridge * np.eye(len(self.nodes)),
            homogeneous.T @ differences)
        residual = differences - homogeneous @ initial
        objective = np.sum(residual**2, axis=0) + initial_ridge * np.sum(initial**2, axis=0)
        return pd.DataFrame(dict(condition=self.condition_names, objective=objective,
            score=-objective, observed_values=int(valid.sum())))

    def save(self, path):
        np.savez_compressed(path, schema=np.array('conditions_v1'), nodes=np.array(self.nodes),
            condition_names=np.array(self.condition_names), context_names=np.array(self.context_names),
            null_condition=np.array(self.null_condition), reference_context=np.array(self.reference_context),
            support=self.support, matrix=self.matrix, condition_input=self.condition_input,
            context_input=self.context_input, offset=self.offset, mean=self.mean, scale=self.scale,
            metadata=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['schema']) != 'conditions_v1':
                raise ValueError('Incorrect condition model schema')
            return cls(tuple(saved['nodes']), tuple(saved['condition_names']), tuple(saved['context_names']),
                str(saved['null_condition']), str(saved['reference_context']), saved['support'], saved['matrix'],
                saved['condition_input'], saved['context_input'], saved['offset'], saved['mean'], saved['scale'],
                json.loads(str(saved['metadata'])))


def fit_conditions(states, times, nodes, condition_indices, context_indices, condition_names, context_names,
                   support, ridge=.1, null_condition='DMSO', reference_context='PBS', stability_margin=1e-4):
    """Fit A/B/C/d on observed real-time increments, with unpenalized intercept."""
    states = np.asarray(states, dtype=float)
    p = len(nodes)
    if states.ndim != 3 or states.shape[1:] != (len(times), p) or ridge <= 0:
        raise ValueError('Invalid training dimensions or ridge')
    times = np.asarray(times, dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid sampling times')
    mean = np.nanmean(states, axis=(0, 1))
    scale = np.nanstd(states, axis=(0, 1)) + 1e-6
    z = (states - mean) / scale
    left, right = z[:, :-1], z[:, 1:]
    x = (.5 * (left + right)).reshape(-1, p)
    y = ((right - left) / np.diff(times)[None, :, None]).reshape(-1, p)
    active_condition = [i for i, name in enumerate(condition_names) if name != null_condition]
    active_context = [i for i, name in enumerate(context_names) if name != reference_context]
    u = np.eye(len(condition_names))[condition_indices][:, active_condition]
    v = np.eye(len(context_names))[context_indices][:, active_context]
    inputs = np.repeat(np.column_stack([u, v, np.ones(len(states))]), len(times) - 1, axis=0)
    valid = np.isfinite(x).all(1) & np.isfinite(y).all(1)
    x, y, inputs = x[valid], y[valid], inputs[valid]
    if len(x) < 2:
        raise ValueError('Insufficient complete observed transitions')
    a = np.zeros((p, p))
    parameters = np.zeros((p, inputs.shape[1]))
    for row in range(p):
        columns = np.flatnonzero(support[row] | (np.arange(p) == row))
        design = np.column_stack([x[:, columns], inputs])
        penalty = np.eye(design.shape[1]) * ridge
        penalty[-1, -1] = 0.
        coef = np.linalg.solve(design.T @ design / len(x) + penalty, design.T @ y[:, row] / len(x))
        a[row, columns] = coef[:len(columns)]
        parameters[row] = coef[len(columns):]
    before = float(np.max(np.linalg.eigvals(a).real))
    shift = max(0., before + stability_margin) if stability_margin is not None else 0.
    if shift:
        a -= shift * np.eye(p)
        penalty = np.eye(inputs.shape[1]) * ridge
        penalty[-1, -1] = 0.
        parameters = np.linalg.solve(inputs.T @ inputs / len(x) + penalty,
            inputs.T @ (y - x @ a.T) / len(x)).T
    b = np.zeros((p, len(condition_names)))
    c = np.zeros((p, len(context_names)))
    b[:, active_condition] = parameters[:, :len(active_condition)]
    c[:, active_context] = parameters[:, len(active_condition):-1]
    return ConditionDynamics(tuple(nodes), tuple(condition_names), tuple(context_names), null_condition,
        reference_context, np.asarray(support, dtype=bool), a, b, c, parameters[:, -1], mean, scale,
        dict(ridge=ridge, valid_training_transitions=len(x), dropped_training_transitions=int((~valid).sum()),
            stability_margin_per_time_unit=stability_margin, spectral_abscissa_before=before,
            diagonal_shift=shift, idop_weights_used=False, numerical_input_estimation='joint ridge; unpenalized intercept'))
