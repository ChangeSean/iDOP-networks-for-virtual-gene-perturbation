"""Propagation parameters estimated independently of IDOP edge-selection coefficients."""
from dataclasses import dataclass,field,asdict
import json
import numpy as np
import pandas as pd
from scipy.linalg import expm
from .data import names,time_vector


@dataclass(frozen=True)
class PropagationConfig:
    backend: str = 'signed'
    input_mode: str = 'persistent'
    ridge: float = .1
    derivative: str = 'trapezoid'
    stability_margin: float | None = .01
    heat_rate: float = 1.
    heat_decay: float = .1

    def __post_init__(self):
        if self.backend not in ['signed','heat']:raise ValueError('backend must be signed or heat')
        if self.input_mode not in ['persistent','none']:raise ValueError('input_mode must be persistent or none')
        if self.derivative not in ['left','trapezoid']:raise ValueError('Unknown derivative design')
        if not np.isfinite(self.ridge) or self.ridge<=0:raise ValueError('ridge must be positive')
        if self.stability_margin is not None and (not np.isfinite(self.stability_margin) or self.stability_margin<0):
            raise ValueError('Invalid stability margin')
        if not np.isfinite(self.heat_rate) or not np.isfinite(self.heat_decay) or min(self.heat_rate,self.heat_decay)<0:
            raise ValueError('Invalid heat parameters')


def unweighted_generator(mask,rate=1.,decay=.1):
    """Directed, degree-normalized binary random-walk generator; columns are sources."""
    mask=np.asarray(mask,dtype=bool);p=len(mask)
    if mask.shape!=(p,p) or np.diag(mask).any():raise ValueError('Expected square cross-node mask')
    if not np.isfinite(rate) or not np.isfinite(decay) or rate<0 or decay<0:raise ValueError('Invalid rates')
    degree=mask.sum(0);active=degree>0
    transition=mask.astype(float)/np.maximum(degree,1)[None,:]
    return rate*(transition-np.diag(active.astype(float)))-decay*np.eye(p)


@dataclass
class LinearPropagation:
    nodes: tuple
    support: np.ndarray
    matrix: np.ndarray
    forcing: np.ndarray
    trained_sources: np.ndarray
    metadata: dict
    _cache: dict = field(default_factory=dict,repr=False)

    def __post_init__(self):
        self.nodes=names(self.nodes,'nodes');p=len(self.nodes)
        self.support=np.array(self.support,dtype=bool,copy=True)
        self.matrix=np.array(self.matrix,dtype=float,copy=True)
        self.forcing=np.array(self.forcing,dtype=float,copy=True)
        self.trained_sources=np.array(self.trained_sources,dtype=bool,copy=True)
        if self.support.shape!=(p,p) or self.matrix.shape!=(p,p) or self.forcing.shape!=(p,) or self.trained_sources.shape!=(p,):
            raise ValueError('Inconsistent propagation shapes')
        if np.diag(self.support).any():raise ValueError('Self terms belong to dynamics, not cross-node topology')
        if not np.isfinite(self.matrix).all() or not np.isfinite(self.forcing).all():raise ValueError('Nonfinite propagation parameters')
        forbidden=~(self.support|np.eye(p,dtype=bool))
        if np.any(np.abs(self.matrix[forbidden])>1e-12):raise ValueError('Dynamics contains an edge outside the IDOP support')
        # Prevent stale exponential caches after direct array mutation.
        for array in [self.support,self.matrix,self.forcing,self.trained_sources]:array.flags.writeable=False

    def node_index(self,node):
        if isinstance(node,str):
            if node not in self.nodes:raise ValueError(f'Unknown node {node}')
            return self.nodes.index(node)
        if not isinstance(node,(int,np.integer)) or not -1<=node<len(self.nodes):raise ValueError('Invalid node index')
        return int(node)

    def operators(self,dt):
        dt=float(dt)
        if not np.isfinite(dt) or dt<0:raise ValueError('Prediction duration must be finite and nonnegative')
        if dt not in self._cache:
            p=len(self.nodes);block=np.zeros((2*p,2*p));block[:p,:p]=self.matrix;block[:p,p:]=np.eye(p)
            transition=expm(block*dt)
            if not np.isfinite(transition).all():raise ValueError('Propagation overflow over requested duration')
            self._cache[dt]=(transition[:p,:p],transition[:p,p:])
        return self._cache[dt]

    def forecast(self,initial,times,source=None,amplitude=1.,initial_time=0.):
        times=time_vector(times);initial=np.asarray(initial,dtype=float)
        if initial.shape!=(len(self.nodes),) or not np.isfinite(initial).all():raise ValueError('Initial state must be finite, one value per node')
        if not np.isfinite(initial_time) or times[0]<initial_time:raise ValueError('Requested time precedes initial state')
        if not np.isfinite(amplitude):raise ValueError('Nonfinite input amplitude')
        u=np.zeros(len(self.nodes));persistent=self.metadata.get('input_mode','none')=='persistent'
        if persistent and source is None and amplitude!=0:raise ValueError('Persistent forecast requires source identity or amplitude=0')
        if source is not None:
            index=self.node_index(source)
            if not persistent and index!=-1:raise ValueError('This model was fitted without a persistent input')
            if index!=-1 and amplitude!=0:
                if not self.trained_sources[index]:raise ValueError('Persistent input was not trained for this source')
                u[index]=self.forcing[index]*amplitude
        result=[]
        for t in times:
            phi,integral=self.operators(t-initial_time);result.append(phi@initial+integral@u)
        return np.array(result)

    def parameter_table(self):
        p=len(self.nodes)
        return pd.DataFrame([dict(source=self.nodes[j],target=self.nodes[i],
            propagation_coefficient=float(self.matrix[i,j]),role='self_dynamics' if i==j else 'cross_node',
            parameter_origin=self.metadata.get('parameter_origin','unspecified'))
            for i,j in np.argwhere(self.support|np.eye(p,dtype=bool))])

    def save(self,path):
        np.savez_compressed(path,schema=np.array(1),nodes=np.array(self.nodes),support=self.support,
            matrix=self.matrix,forcing=self.forcing,trained_sources=self.trained_sources,
            metadata=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as s:
            if int(s['schema'])!=1:raise ValueError('Unsupported model schema')
            return cls(tuple(s['nodes']),s['support'],s['matrix'],s['forcing'],s['trained_sources'],json.loads(str(s['metadata'])))


def fit_propagation(batch,topology,config=PropagationConfig()):
    """Estimate dz/dt=A z + diag(b)u under binary support, using only training trajectories.

    Trapezoid design uses observed adjacent state averages and actual elapsed time.
    A and b are new dynamical estimates; IDOP coefficients/frequencies are not inputs.
    """
    if tuple(batch.nodes)!=tuple(topology.nodes):raise ValueError('Topology and training node orders differ')
    p=len(batch.nodes);n,nt,_=batch.states.shape
    if config.input_mode=='persistent' and batch.source_indices is None:
        raise ValueError('Persistent-input training requires source_indices (-1 denotes control)')
    scale=np.maximum(np.sqrt(np.mean(batch.states**2,axis=(0,1))),.02)
    z=batch.states/scale
    left=z[:,:-1];right=z[:,1:];dt=np.diff(batch.times)
    x=(.5*(left+right) if config.derivative=='trapezoid' else left).reshape(-1,p)
    y=((right-left)/dt[None,:,None]).reshape(-1,p)
    src=np.repeat(batch.source_indices,nt-1) if batch.source_indices is not None else np.full(n*(nt-1),-1)
    amp=np.repeat(batch.input_amplitudes,nt-1)
    trained=np.array([np.any((src==i)&(amp!=0)) for i in range(p)]) if config.input_mode=='persistent' else np.zeros(p,dtype=bool)
    forcing=np.zeros(p);a=np.zeros((p,p));before=0.;shift=0.
    if config.backend=='signed':
        for target in range(p):
            allowed=np.flatnonzero(topology.mask[target]|(np.arange(p)==target))
            indicator=(src==target).astype(float)*amp
            include=config.input_mode=='persistent' and trained[target]
            design=np.column_stack([x[:,allowed],indicator]) if include else x[:,allowed]
            coef=np.linalg.solve(design.T@design/len(x)+config.ridge*np.eye(design.shape[1]),design.T@y[:,target]/len(x))
            a[target,allowed]=coef[:len(allowed)]
            if include:forcing[target]=coef[-1]*scale[target]
        a=a*scale[:,None]/scale[None,:]
        before=float(np.max(np.linalg.eigvals(a).real))
        if config.stability_margin is not None:
            shift=max(0.,before+config.stability_margin);a-=shift*np.eye(p)
            # Re-estimate the source-local input after the explicit diagonal projection.
            physical_x=x*scale;physical_y=y*scale
            residual=physical_y-physical_x@a.T
            for target in np.flatnonzero(trained):
                indicator=(src==target).astype(float)*amp
                forcing[target]=(indicator@residual[:,target]/len(x))/(indicator@indicator/len(x)+config.ridge)
        origin='independently fitted real-time state increments under binary IDOP support'
    else:
        a=unweighted_generator(topology.mask,config.heat_rate,config.heat_decay)
        residual=y*scale-(x*scale)@a.T
        for target in np.flatnonzero(trained):
            indicator=(src==target).astype(float)*amp
            forcing[target]=(indicator@residual[:,target]/len(x))/(indicator@indicator/len(x)+config.ridge)
        before=float(np.max(np.linalg.eigvals(a).real))
        origin='binary adjacency plus global heat rate/decay; no IDOP numerical weights'
    return LinearPropagation(batch.nodes,topology.mask,a,forcing,trained,dict(
        **asdict(config),parameter_origin=origin,matrix_order='target,source',
        self_terms='estimated or prescribed separately; not IDOP edges',
        spectral_abscissa_before_projection=before,diagonal_stability_shift=shift,
        spectral_abscissa=float(np.max(np.linalg.eigvals(a).real)),training_times=batch.times.tolist(),
        training_trajectories=n,training_transitions=len(x),idop_weights_used=False,
        power_law_fitting=False,proximal=False,
        interpretation='effective response propagation, not proof of direct biochemical regulation'))
