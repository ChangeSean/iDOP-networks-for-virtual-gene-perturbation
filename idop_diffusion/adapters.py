"""Existing GEO count-processing outputs -> actual-time guide/population batches."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import TimeSeriesBatch


def load_gene_perturbation_batches(project_root,strict=False,min_cells=10):
    folder=Path(project_root)/'data/GSE213069/processed'
    protocol=json.loads((folder/'preprocessing_protocol.json').read_text())
    nodes=tuple(protocol['panel']);targets=protocol['targets'];rule='pass_strict_qc' if strict else 'pass_qc'
    data={};counts={};control_counts={};controls={}
    for day in [2,3,4,5]:
        meta=pd.read_csv(folder/f'cell_metadata_day{day}.csv.gz')
        with np.load(folder/f'panel_day{day}.npz',allow_pickle=False) as s:
            if tuple(s['genes'])!=nodes or not np.array_equal(s['barcodes'],meta.barcode):
                raise ValueError('Cell/node order mismatch in processed inputs')
            values=s['log1p_cp10k'].astype(float)
        ok=meta[rule].to_numpy(bool);ctrl=ok&(meta.target.to_numpy()=='CONTROL')
        if ctrl.sum()<min_cells:raise ValueError('Too few control cells')
        controls[day]=values[ctrl].mean(0);control_counts[day]=int(ctrl.sum())
        for arm in [1,2]:
            for gene in targets:
                take=ok&(meta.top_guide.to_numpy()==f'{gene}_{arm}')
                key=(arm,gene,day);counts[key]=int(take.sum())
                if counts[key]>=min_cells:data[key]=values[take].mean(0)-controls[day]
    # A common, count-defined evaluation cohort, not response-selected features.
    common=[g for g in targets if all(counts[(a,g,d)]>=min_cells for a in [1,2] for d in [2,3,4,5])]
    if len(common)<2:raise ValueError('Too few complete target trajectories')
    batches={a:TimeSeriesBatch(np.array([[data[(a,g,d)] for d in [2,3,4,5]] for g in common]),
        np.array([2.,3.,4.,5.]),nodes,np.array([nodes.index(g) for g in common]),
        trajectory_ids=tuple(f'{g}_guide{a}' for g in common)) for a in [1,2]}
    metadata=dict(dataset='GSE213069',quality_rule=rule,min_cells=min_cells,targets=common,
        all_targets=targets,markers=[g for g in nodes if g not in targets],controls='all known controls shared within experiment',
        control_cell_counts=control_counts,normalization='mean single-cell log1p CP10K minus same-day control mean',
        unit='target/guide/day population, not paired cells or biological replicates',
        cohort_selection='cell counts only, common across guides; response values not used to select cohort')
    return batches,metadata
