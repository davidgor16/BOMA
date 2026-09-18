"""Evaluate saved predictions with the archived metric functions and explicit settings."""
from pathlib import Path
import argparse
import ast
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ASPECTS=['P-ACC','W-ACC','W-STR','W-TOT','U-ACC','U-COM','U-FLU','U-PRO','U-TOT']

def load_functions(path,names):
    tree=ast.parse(path.read_bytes(),filename=str(path))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError(f'Missing metric functions in {path}')
    scope={'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True,help='One experiment directory containing run/fold predictions.')
    parser.add_argument('--output',type=Path,required=True,help='New JSON file outside the input experiment.')
    parser.add_argument('--bins',choices=['uniform5','binary-special'],required=True,help='Choose and record the evaluation convention explicitly.')
    parser.add_argument('--min-support',type=int,required=True,help='Macro-MSE minimum bin count; archived script uses 5, manuscript describes 1.')
    parser.add_argument('--iterations',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--scale',type=float,default=5.)
    args=parser.parse_args()
    source=args.experiment.resolve();output=args.output.resolve()
    if not source.is_dir():parser.error('Experiment directory does not exist. Restore its artifact archive first.')
    if output.exists() or output.is_relative_to(source):parser.error('Output must be new and outside the input experiment.')
    if args.min_support<1 or args.iterations<1:parser.error('Support and iterations must be positive.')
    for model in ROOT.glob('0*'):
        if any(output.is_relative_to(model/x) for x in ['exp','data','exp_hierTFR']):parser.error('Output cannot overlap archived data or experiments.')
    paths=sorted(source.rglob('phn_pred.npy'))
    if not paths:parser.error('No prediction sets found.')
    ccc=load_functions(ROOT/'models_analysis/calculate_ccc_bootstrap_balanced.py',['ccc_score','bootstrap_adaptive_balanced_ccc'])
    mse=load_functions(ROOT/'models_analysis/calculate_micro_macro_mse.py',['calc_macro_mse'])
    np.random.seed(args.seed)
    rows=[]
    for path in paths:
        directory=path.parent
        target={k:np.load(directory/f'{k}_target.npy',allow_pickle=False) for k in ['phn','word','utt']}
        pred={k:np.load(directory/f'{k}_pred.npy',allow_pickle=False) for k in ['phn','word','utt']}
        pairs=[(target['phn'].flatten(),pred['phn'].flatten())]
        for level,columns in [('word',3),('utt',5)]:
            if target[level].ndim!=2 or pred[level].ndim!=2 or target[level].shape[1]!=columns or pred[level].shape[1]!=columns:raise ValueError(f'Unexpected {level} shape in {directory}')
            pairs.extend((target[level][:,i],pred[level][:,i]) for i in range(columns))
        for name,(gt,estimate) in zip(ASPECTS,pairs):
            if gt.shape!=estimate.shape:raise ValueError(f'{directory}, {name}: target {gt.shape} != prediction {estimate.shape}; no automatic alignment or truncation is performed.')
            mask=gt>=0;gt=gt[mask]*args.scale;estimate=estimate[mask]*args.scale
            if not len(gt) or not np.isfinite(gt).all() or not np.isfinite(estimate).all():raise ValueError(f'Empty or nonfinite arrays: {directory}, {name}')
            edges=[7.5] if args.bins=='binary-special' and name in ['W-STR','U-COM'] else [2.,4.,6.,8.]
            boundaries=[0.,*edges,10.];bins=list(zip(boundaries[:-1],boundaries[1:]))+[(0.,10.)]
            balanced,n=ccc['bootstrap_adaptive_balanced_ccc'](gt,estimate,bins,n_iterations=args.iterations,min_power=20)
            rows.append({'run':str(directory.relative_to(source)),'aspect':name,'n':len(gt),'pcc':float(np.corrcoef(gt,estimate)[0,1]),'ccc':float(ccc['ccc_score'](gt,estimate)),'ccc_bs':float(balanced),'bootstrap_n_per_bin':float(n),'micro_mse':float(np.mean((gt-estimate)**2)),'macro_mse':mse['calc_macro_mse'](estimate,gt,edges,min_support=args.min_support)})
        print(f'Evaluated {directory.relative_to(source)}',flush=True)
    summary={}
    for name in ASPECTS:
        selected=[r for r in rows if r['aspect']==name]
        summary[name]={metric:{'mean':float(np.nanmean([r[metric] for r in selected])),'std_population':float(np.nanstd([r[metric] for r in selected])),'std_sample':float(np.nanstd([r[metric] for r in selected],ddof=1)) if len(selected)>1 else None} for metric in ['pcc','ccc','ccc_bs','micro_mse','macro_mse']}
    result={'protocol':{'bins':args.bins,'macro_min_support':args.min_support,'ccc_intervals':'right-closed; first interval includes zero','mse_intervals':'np.digitize default: left-closed','bootstrap_iterations':args.iterations,'bootstrap_min_power':20,'seed':args.seed,'scale':args.scale,'numpy':np.__version__},'runs':len(paths),'results':rows,'summary':summary}
    def finite_json(value):
        if isinstance(value,float) and not np.isfinite(value):return None
        if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
        if isinstance(value,list):return [finite_json(v) for v in value]
        return value
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(finite_json(result),stream,indent=2,allow_nan=False)
    print(f'Saved {output}')

if __name__=='__main__':main()
