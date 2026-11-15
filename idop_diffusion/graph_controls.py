"""Directed degree-preserving control graphs for matched propagation experiments."""
import hashlib
import numpy as np
from .topology import Topology


def rewire_support(mask, seed, swaps_per_edge=10):
    original = np.asarray(mask, dtype=bool)
    result = original.copy()
    edges = np.argwhere(result)
    rng = np.random.default_rng(seed)
    accepted = 0
    target = swaps_per_edge * len(edges)
    for attempt in range(max(1, target * 40)):
        if accepted >= target or len(edges) < 2:
            break
        first, second = rng.choice(len(edges), 2, replace=False)
        target1, source1 = edges[first]
        target2, source2 = edges[second]
        if target1 == target2 or source1 == source2 or target1 == source2 or target2 == source1:
            continue
        if result[target1, source2] or result[target2, source1]:
            continue
        result[target1, source1] = result[target2, source2] = False
        result[target1, source2] = result[target2, source1] = True
        edges[first] = [target1, source2]
        edges[second] = [target2, source1]
        accepted += 1
    np.testing.assert_array_equal(result.sum(0), original.sum(0))
    np.testing.assert_array_equal(result.sum(1), original.sum(1))
    assert not np.diag(result).any()
    return result, dict(seed=seed, accepted_swaps=accepted, requested_swaps=target,
        changed_entries=int(np.sum(result != original)), sha256=hashlib.sha256(result.tobytes()).hexdigest())


def control_topology(reference, method):
    p = len(reference.nodes)
    audit = dict(method=method)
    if method == 'idop':
        return reference, audit
    if method == 'self_only':
        mask = np.zeros((p, p), dtype=bool)
    elif method == 'dense':
        mask = ~np.eye(p, dtype=bool)
    elif method.startswith('random_'):
        mask, record = rewire_support(reference.mask, 20261106 + int(method.split('_')[1]))
        audit.update(record)
    else:
        raise ValueError(f'Unknown control {method}')
    result = Topology(reference.nodes, mask, mask.astype(float), mask[None],
        np.array([[reference.window_bounds[0, 0], reference.window_bounds[-1, 1]]]),
        dict(selector='matched graph control', **audit))
    return result, audit
