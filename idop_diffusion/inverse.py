"""Candidate source inversion through the same forward propagation model."""
import numpy as np
import pandas as pd
from .data import names,time_vector


def trace_sources(model,observed,times,sensors,candidates=None,kind='persistent',
                  initial_state=None,initial_ridge=1.,amplitude_ridge=0.,amplitude_bounds=(0.,2.)):
    """Rank single sources by regularized trajectory reconstruction, without source labels.

    For persistent input, an unknown anchor state is a common nuisance variable,
    estimated with the same ridge penalty for every candidate. This avoids reading
    hidden target expression. Pulse inversion requires an explicit known initial
    baseline: a free initial state would absorb the entire pulse and destroy identification.
    Times are elapsed from the supplied/unknown anchor state; onset is not inferred.
    Missing sensor observations may be NaN, but infinity is rejected.
    """
    if kind not in ['persistent','pulse']:raise ValueError('kind must be persistent or pulse')
    times=time_vector(times)
    if times[0]<0:raise ValueError('Elapsed observation times must be nonnegative')
    sensors=names(sensors,'sensors');candidates=model.nodes if candidates is None else names(candidates,'candidates')
    si=np.array([model.node_index(v) for v in sensors]);ci=np.array([model.node_index(v) for v in candidates])
    y=np.asarray(observed,dtype=float)
    if y.shape!=(len(times),len(sensors)) or np.isinf(y).any():raise ValueError('Incorrect/nonfinite observed sensor matrix')
    take=np.isfinite(y).ravel()
    if take.sum()<2:raise ValueError('At least two observed sensor values are required')
    if kind=='persistent' and model.metadata.get('input_mode')!='persistent':raise ValueError('Persistent inputs were not fitted')
    if kind=='pulse' and initial_state is None:raise ValueError('Pulse inversion needs a known initial baseline; free initial states absorb any pulse')
    if not np.isfinite(initial_ridge) or initial_ridge<=0 or not np.isfinite(amplitude_ridge) or amplitude_ridge<0:
        raise ValueError('initial_ridge must be positive and amplitude_ridge nonnegative')
    low,high=amplitude_bounds
    low=-np.inf if low is None else float(low);high=np.inf if high is None else float(high)
    if np.isnan(low) or np.isnan(high) or low>high:raise ValueError('Invalid amplitude bounds')
    hs=[];ss=[]
    for t in times:
        phi,integral=model.operators(t);hs.append(phi[si])
        ss.append((integral[si][:,ci]*model.forcing[ci][None,:]) if kind=='persistent' else phi[si][:,ci])
    h=np.vstack(hs)[take];s=np.vstack(ss)[take];y=y.ravel()[take]
    if initial_state is None:
        gram=h.T@h+initial_ridge*np.eye(len(model.nodes))
        nuisance_y=np.linalg.solve(gram,h.T@y);nuisance_s=np.linalg.solve(gram,h.T@s)
        projected_y=y-h@nuisance_y;projected_s=s-h@nuisance_s
        numerator=s.T@projected_y
        denominator=np.sum(s*projected_s,axis=0)+amplitude_ridge
    else:
        baseline=np.asarray(initial_state,dtype=float)
        if baseline.shape!=(len(model.nodes),) or not np.isfinite(baseline).all():raise ValueError('Invalid initial_state')
        y=y-h@baseline;numerator=s.T@y;denominator=np.sum(s*s,axis=0)+amplitude_ridge
    rows=[]
    for j,(label,index) in enumerate(zip(candidates,ci)):
        available=kind=='pulse' or bool(model.trained_sources[index])
        if not available:
            rows.append(dict(source=label,available=False,status='untrained_source',amplitude=np.nan,
                squared_error=np.inf,objective=np.inf,score=-np.inf));continue
        informative=denominator[j]>1e-14
        amplitude=float(np.clip(numerator[j]/denominator[j],low,high)) if informative else float(np.clip(0.,low,high))
        if initial_state is None:
            initial=nuisance_y-amplitude*nuisance_s[:,j]
            residual=y-h@initial-amplitude*s[:,j]
            penalty=initial_ridge*float(initial@initial)
        else:residual=y-amplitude*s[:,j];penalty=0.
        squared=float(residual@residual);objective=squared+penalty+amplitude_ridge*amplitude**2
        rows.append(dict(source=label,available=True,status='ok' if informative else 'uninformative_signature',
            amplitude=amplitude,squared_error=squared,objective=objective,score=-objective))
    result=pd.DataFrame(rows)
    result['rank_min']=np.nan;result['rank_max']=np.nan;result['rank']=np.nan
    good=result.available.to_numpy();values=result.objective.to_numpy()
    for i in np.flatnonzero(good):
        tolerance=1e-10*max(1.,abs(values[i]));better=int(np.sum(good&(values<values[i]-tolerance)))
        tied=int(np.sum(good&(np.abs(values-values[i])<=tolerance)))
        result.loc[i,['rank_min','rank_max','rank']]=[1+better,better+tied,1+better+(tied-1)/2]
    result=result.sort_values(['objective','source'],kind='stable').reset_index(drop=True)
    result.attrs.update(kind=kind,initial_state='ridge-estimated nuisance' if initial_state is None else 'known',
        scores_are_probabilities=False,onset_inferred=False)
    return result
