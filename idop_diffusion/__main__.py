"""python -m idop_diffusion fit|predict|trace --help"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from . import TimeSeriesBatch,EdgeConfig,PropagationConfig,IDOPDiffusion


def main():
    parser=argparse.ArgumentParser(description='IDOP binary graph -> independent propagation -> forecast/source inversion')
    subs=parser.add_subparsers(dest='command',required=True)
    fit=subs.add_parser('fit');fit.add_argument('--input',type=Path,required=True);fit.add_argument('--output',type=Path,required=True)
    fit.add_argument('--alpha',type=float,default=.1);fit.add_argument('--window-points',type=int,default=2,help='0 uses one pooled window')
    fit.add_argument('--threshold',type=float,default=.6);fit.add_argument('--ridge',type=float,default=.1)
    fit.add_argument('--backend',choices=['signed','heat'],default='signed');fit.add_argument('--input-mode',choices=['persistent','none'],default='persistent')
    fit.add_argument('--heat-rate',type=float,default=1.);fit.add_argument('--heat-decay',type=float,default=.1)
    fit.add_argument('--overwrite',action='store_true')
    for command in ['predict','trace']:
        p=subs.add_parser(command);p.add_argument('--model',type=Path,required=True,help='Exported model directory')
        p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--overwrite',action='store_true')
        if command=='trace':
            p.add_argument('--kind',choices=['persistent','pulse'],default='persistent');p.add_argument('--initial-ridge',type=float,default=1.)
            p.add_argument('--signed-amplitude',action='store_true',help='Allow unrestricted signed source amplitudes')
    args=parser.parse_args()
    if args.command=='fit':
        data=TimeSeriesBatch.load(args.input)
        model=IDOPDiffusion.fit(data,EdgeConfig(alpha=args.alpha,window_points=args.window_points or None,threshold=args.threshold),
            PropagationConfig(backend=args.backend,input_mode=args.input_mode,ridge=args.ridge,heat_rate=args.heat_rate,heat_decay=args.heat_decay))
        model.export(args.output,args.overwrite)
        (args.output/'training_input.json').write_text(json.dumps(dict(path=str(args.input.resolve()),
            sha256=hashlib.sha256(args.input.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
        print(json.dumps(dict(nodes=len(data.nodes),edges=int(model.topology.mask.sum()),output=str(args.output))))
        return
    if args.output.exists() and not args.overwrite:raise FileExistsError(args.output)
    if args.command=='predict' and args.output.suffix.lower()!='.npz':raise ValueError('Prediction output must end in .npz')
    model=IDOPDiffusion.load(args.model)
    with np.load(args.input,allow_pickle=False) as data:
        # Query truth labels are never read; trace has no query-source argument.
        times=data['times'];ids=data['query_ids'] if 'query_ids' in data else None
        if args.command=='predict':
            initial=data['initial'];initial=initial[None,:] if initial.ndim==1 else initial
            if initial.ndim!=2 or initial.shape[1]!=len(model.propagator.nodes):raise ValueError('initial must have shape (queries,nodes)')
            if 'nodes' not in data or tuple(data['nodes'])!=model.propagator.nodes:raise ValueError('Query node order must match model exactly')
            sources=data['source_indices'] if 'source_indices' in data else [None]*len(initial)
            amplitudes=data['input_amplitudes'] if 'input_amplitudes' in data else np.ones(len(initial))
            if len(sources)!=len(initial) or len(amplitudes)!=len(initial):raise ValueError('Query input lengths differ')
            initial_time=float(data['initial_time']) if 'initial_time' in data else 0.
            pred=np.array([model.predict(row,times,source=s,amplitude=a,initial_time=initial_time) for row,s,a in zip(initial,sources,amplitudes)])
            ids=np.array([f'query_{i}' for i in range(len(initial))]) if ids is None else ids
            if len(ids)!=len(initial):raise ValueError('Wrong number of query_ids')
            args.output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.output,prediction=pred,times=times,nodes=np.array(model.propagator.nodes),query_ids=ids)
        else:
            observed=data['observed'];observed=observed[None,:,:] if observed.ndim==2 else observed
            if observed.ndim!=3:raise ValueError('observed must have shape (queries,times,sensors)')
            candidates=tuple(data['candidates']) if 'candidates' in data else None
            initial=data['initial'] if 'initial' in data else None
            if initial is not None and initial.shape!=(len(observed),len(model.propagator.nodes)):raise ValueError('initial must have shape (queries,nodes)')
            ids=np.array([f'query_{i}' for i in range(len(observed))]) if ids is None else ids
            if len(ids)!=len(observed):raise ValueError('Wrong number of query_ids')
            rows=[]
            for i,row in enumerate(observed):
                ranks=model.trace(row,times,tuple(data['sensors']),candidates,kind=args.kind,
                    initial_state=None if initial is None else initial[i],initial_ridge=args.initial_ridge,
                    amplitude_bounds=(None,None) if args.signed_amplitude else (0.,2.))
                ranks.insert(0,'query_id',str(ids[i]));rows.append(ranks)
            args.output.parent.mkdir(parents=True,exist_ok=True);pd.concat(rows,ignore_index=True).to_csv(args.output,index=False)
    print(str(args.output))

if __name__=='__main__':main()
