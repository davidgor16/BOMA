"""Check a saved GOPT checkpoint on four real utterances without saving model state."""
from pathlib import Path
import argparse,ast,importlib.util,json,sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();output=a.output.resolve();assets=a.assets_root.resolve()
    if output.exists():p.error('Choose a new output file.')
    if any(output.is_relative_to(assets/'01_gopt'/name) for name in ['data','exp']):p.error('Output cannot be inside archived inputs.')
    torch.set_num_threads(2);torch.manual_seed(0)
    path=ROOT/'01_gopt/src/models/gopt.py'
    spec=importlib.util.spec_from_file_location('archived_gopt',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    train_path=ROOT/'01_gopt/src/traintest_BOMA.py';tree=ast.parse(train_path.read_bytes())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['BalancedMSELoss','GoPDataset']]
    scope={'np':np,'torch':torch,'nn':torch.nn,'Dataset':torch.utils.data.Dataset}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(train_path),'exec'),scope)
    data=assets/'01_gopt/data/seq_data_librispeech'
    x=torch.tensor(np.load(data/'te_feat.npy',mmap_mode='r')[:4].copy(),dtype=torch.float32)
    y=torch.tensor(np.load(data/'te_label_phn.npy',mmap_mode='r')[:4].copy(),dtype=torch.float32)
    x=scope['GoPDataset'].norm_valid(None,x,3.203,4.045).to(a.device);y=y.to(a.device)
    model=module.GOPT(embed_dim=24,num_heads=1,depth=3,input_dim=84).to(a.device)
    checkpoint=assets/'01_gopt/exp/gopt-BOMA/fold_1/models/best_audio_model.pth'
    state=torch.load(checkpoint,map_location=a.device,weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    model.eval();scores=model(x,y[:,:,0]);target=y[:,:,1]
    loss=scope['BalancedMSELoss'](bins=10)(scores[5].squeeze(-1),target,mask=target>=0)
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(t).all() for t in scores):raise AssertionError('Nonfinite outputs or loss.')
    grads=[v.grad for v in model.parameters() if v.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):raise AssertionError('Missing or nonfinite gradients.')
    record={'status':'pass','utterances':4,'device':a.device,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'checkpoint':str(checkpoint),'loss':loss.item(),'output_shapes':[list(t.shape) for t in scores],'optimizer_steps':0,'saved_checkpoints':0,'saved_predictions':0}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
