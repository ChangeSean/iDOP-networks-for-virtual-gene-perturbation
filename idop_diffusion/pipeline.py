"""One trained binary IDOP graph, one independent propagator, two downstream tasks."""
from pathlib import Path
import hashlib,json,platform
import numpy as np
import pandas as pd
import scipy,sklearn
from .topology import EdgeConfig,Topology,select_topology
from .propagation import PropagationConfig,LinearPropagation,fit_propagation
from .inverse import trace_sources


class IDOPDiffusion:
    def __init__(self,topology,propagator):
        if topology.nodes!=propagator.nodes or not np.array_equal(topology.mask,propagator.support):
            raise ValueError('Topology and propagation support differ')
        self.topology=topology;self.propagator=propagator

    @classmethod
    def fit(cls,training,edges=EdgeConfig(),propagation=PropagationConfig()):
        topology=select_topology(training,edges)
        return cls(topology,fit_propagation(training,topology,propagation))

    def predict(self,*args,**kwargs):return self.propagator.forecast(*args,**kwargs)
    def trace(self,*args,**kwargs):return trace_sources(self.propagator,*args,**kwargs)

    def export(self,directory,overwrite=False):
        directory=Path(directory)
        if directory.exists() and any(directory.iterdir()) and not overwrite:
            raise FileExistsError(f'Output exists: {directory}; use a new path or overwrite=True')
        directory.mkdir(parents=True,exist_ok=True)
        self.topology.save(directory/'topology.npz');self.propagator.save(directory/'propagation.npz')
        self.topology.edge_table()[['source','target']].to_csv(directory/'network_edges.csv',index=False)
        self.topology.edge_table(False).to_csv(directory/'selection_evidence.csv',index=False)
        self.propagator.parameter_table().to_csv(directory/'propagation_parameters.csv',index=False)
        pd.DataFrame(dict(node=self.propagator.nodes,input_coefficient=self.propagator.forcing,
            trained_source=self.propagator.trained_sources)).to_csv(directory/'input_parameters.csv',index=False)
        manifest=dict(framework='IDOP binary topology -> independent propagation -> forecast/source inversion',
            schema=1,nodes=list(self.topology.nodes),directed_candidate_edges=int(self.topology.mask.sum()),
            selection=self.topology.metadata,propagation=self.propagator.metadata,
            runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                scipy=scipy.__version__,sklearn=sklearn.__version__),
            package_sha256={str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.rglob('*.py'))},
            artifacts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())
                if p.is_file() and p.name in ['topology.npz','propagation.npz','network_edges.csv','selection_evidence.csv',
                    'propagation_parameters.csv','input_parameters.csv']})
        (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

    @classmethod
    def load(cls,directory):
        directory=Path(directory)
        return cls(Topology.load(directory/'topology.npz'),LinearPropagation.load(directory/'propagation.npz'))
