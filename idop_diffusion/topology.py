"""IDOP selects binary support only, from actual-time windows of training states."""
from dataclasses import dataclass,asdict
import json,warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from .data import names
from ._vendor.idop_edge_select import edge_select


@dataclass(frozen=True)
class EdgeConfig:
    alpha: float = .1
    window_points: int | None = 2
    stride: int = 1
    threshold: float = .6
    min_rows: int = 4

    def __post_init__(self):
        if not np.isfinite(self.alpha) or self.alpha<=0:raise ValueError('alpha must be positive')
        if self.window_points is not None and (not isinstance(self.window_points,int) or self.window_points<2):
            raise ValueError('window_points must be at least 2, or None for pooled selection')
        if not isinstance(self.stride,int) or self.stride<1:raise ValueError('stride must be a positive integer')
        if not np.isfinite(self.threshold) or not 0<=self.threshold<=1:raise ValueError('threshold outside [0,1]')
        if self.min_rows<2:raise ValueError('min_rows must be at least 2')


@dataclass
class Topology:
    nodes: tuple
    mask: np.ndarray
    frequency: np.ndarray
    window_masks: np.ndarray
    window_bounds: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.nodes=names(self.nodes,'nodes');p=len(self.nodes)
        self.mask=np.array(self.mask,dtype=bool,copy=True)
        self.frequency=np.array(self.frequency,dtype=float,copy=True)
        self.window_masks=np.array(self.window_masks,dtype=bool,copy=True)
        self.window_bounds=np.array(self.window_bounds,dtype=float,copy=True)
        if self.mask.shape!=(p,p) or self.frequency.shape!=(p,p):raise ValueError('Incorrect topology shape')
        if np.diag(self.mask).any():raise ValueError('Topology stores cross-node edges only')
        if not np.isfinite(self.frequency).all() or np.any((self.frequency<0)|(self.frequency>1)):
            raise ValueError('Invalid selection frequency')
        if self.window_masks.ndim!=3 or self.window_masks.shape[1:]!=(p,p) or self.window_bounds.shape!=(len(self.window_masks),2):
            raise ValueError('Invalid temporal-window evidence')

    def edge_table(self,selected_only=True):
        records=[]
        for target,source in np.argwhere(~np.eye(len(self.nodes),dtype=bool)):
            if not selected_only or self.mask[target,source]:
                records.append(dict(source=self.nodes[source],target=self.nodes[target],
                    selected=bool(self.mask[target,source]),selection_frequency=float(self.frequency[target,source])))
        return pd.DataFrame(records,columns=['source','target','selected','selection_frequency'])

    def save(self,path):
        np.savez_compressed(path,nodes=np.array(self.nodes),mask=self.mask,frequency=self.frequency,
            window_masks=self.window_masks,window_bounds=self.window_bounds,metadata=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as s:
            return cls(tuple(s['nodes']),s['mask'],s['frequency'],s['window_masks'],s['window_bounds'],json.loads(str(s['metadata'])))


def select_topology(batch,config=EdgeConfig()):
    """Pass observed states to unmodified IDOP nodewise selection; never fit curves.

    Each window contains the same real sampling times from every training trajectory.
    IDOP k=1 selects within that temporal window. Strict frequency > threshold matches
    upstream semantics. Nodewise support is a candidate graph, not causal orientation.
    """
    p=len(batch.nodes);nt=len(batch.times)
    if p<2:raise ValueError('IDOP requires at least two nodes')
    if 'time' in batch.nodes:raise ValueError('Node name time is reserved by IDOP')
    width=nt if config.window_points is None else config.window_points
    if width>nt:raise ValueError('window_points exceeds available sampling times')
    starts=list(range(0,nt-width+1,config.stride))
    if starts[-1]!=nt-width:starts.append(nt-width)
    mapping={g:i for i,g in enumerate(batch.nodes)};masks=[];bounds=[];warning_count=0
    for start in starts:
        values=batch.states[:,start:start+width].transpose(1,0,2).reshape(-1,p)
        if len(values)<config.min_rows:raise ValueError('Too few observed rows in a time window')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            table=edge_select(pd.DataFrame(values,columns=batch.nodes),alpha=config.alpha,k=1,threshold=0.)
        warning_count+=sum(isinstance(w.message,ConvergenceWarning) for w in caught)
        mask=np.zeros((p,p),dtype=bool)
        for row in table.itertuples(index=False):
            for source in row.source.strip('{}').split(','):
                if source:mask[mapping[row.target],mapping[source]]=True
        masks.append(mask);bounds.append([batch.times[start],batch.times[start+width-1]])
    masks=np.array(masks);frequency=masks.mean(0);support=frequency>config.threshold
    np.fill_diagonal(support,False)
    return Topology(batch.nodes,support,frequency,masks,np.array(bounds),dict(
        selector='existing IDOP edge_select, unchanged within each actual-time window',
        config=asdict(config),window_axis='actual sampling time, never concatenated row position',
        matrix_order='target,source',weight_policy='binary support only; selection frequency is evidence, never a propagation coefficient',
        curve_fitting=False,niche_reordering=False,proximal=False,causal_orientation_claim=False,
        training_trajectories=len(batch.states),training_times=batch.times.tolist(),
        convergence_warnings=warning_count))
