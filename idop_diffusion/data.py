"""Actual-time population trajectories. A trajectory never concatenates different cells."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def names(values, label):
    result=tuple(str(v) for v in values)
    if not result or len(set(result))!=len(result) or any(not v or any(c in v for c in ',{}') for v in result):
        raise ValueError(f'{label} must be nonempty, unique names without comma/braces')
    return result


def time_vector(values, minimum=1):
    result=np.asarray(values,dtype=float)
    if result.ndim!=1 or len(result)<minimum or not np.isfinite(result).all() or np.any(np.diff(result)<=0):
        raise ValueError('Times must be a finite, strictly increasing one-dimensional vector')
    return result


@dataclass
class TimeSeriesBatch:
    states: np.ndarray
    times: np.ndarray
    nodes: tuple
    source_indices: np.ndarray | None = None
    input_amplitudes: np.ndarray | None = None
    trajectory_ids: tuple | None = None

    def __post_init__(self):
        self.states=np.array(self.states,dtype=float,copy=True)
        self.times=time_vector(self.times,minimum=2).copy();self.nodes=names(self.nodes,'nodes')
        if self.states.ndim!=3 or self.states.shape[0]<1 or self.states.shape[1:]!=(len(self.times),len(self.nodes)):
            raise ValueError('states must have shape (trajectories, times, nodes)')
        if not np.isfinite(self.states).all():raise ValueError('Training states contain nonfinite values')
        n=self.states.shape[0]
        if self.source_indices is not None:
            raw=np.asarray(self.source_indices)
            if raw.shape!=(n,) or not np.all(np.isfinite(raw)) or not np.all(raw==np.floor(raw)):
                raise ValueError('source_indices must contain one integer per trajectory')
            self.source_indices=raw.astype(int)
            if np.any((self.source_indices < -1)|(self.source_indices>=len(self.nodes))):
                raise ValueError('Source indices must be node indices, or -1 for no source')
        if self.input_amplitudes is None:self.input_amplitudes=np.ones(n)
        else:self.input_amplitudes=np.asarray(self.input_amplitudes,dtype=float).copy()
        if self.input_amplitudes.shape!=(n,) or not np.isfinite(self.input_amplitudes).all():
            raise ValueError('input_amplitudes must be finite, one per trajectory')
        if self.trajectory_ids is None:self.trajectory_ids=tuple(str(i) for i in range(n))
        else:self.trajectory_ids=names(self.trajectory_ids,'trajectory_ids')
        if len(self.trajectory_ids)!=n:raise ValueError('Incorrect trajectory_ids length')

    def take(self,indices):
        indices=np.asarray(indices,dtype=int)
        return TimeSeriesBatch(self.states[indices],self.times,self.nodes,
            None if self.source_indices is None else self.source_indices[indices],self.input_amplitudes[indices],
            tuple(self.trajectory_ids[i] for i in indices))

    def save(self,path):
        values=dict(states=self.states,times=self.times,nodes=np.array(self.nodes),
            input_amplitudes=self.input_amplitudes,trajectory_ids=np.array(self.trajectory_ids))
        if self.source_indices is not None:values['source_indices']=self.source_indices
        np.savez_compressed(path,**values)

    @classmethod
    def load(cls,path):
        with np.load(Path(path),allow_pickle=False) as s:
            return cls(s['states'],s['times'],tuple(s['nodes']),
                s['source_indices'] if 'source_indices' in s else None,
                s['input_amplitudes'] if 'input_amplitudes' in s else None,
                tuple(s['trajectory_ids']) if 'trajectory_ids' in s else None)
