"""Run one archived training entry point with a fresh output directory."""
from pathlib import Path
import argparse,json,os,shlex,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',choices=['gopt','hipama','hiertfr','conpco','hmamba','m3c'])
    p.add_argument('--variant',choices=['server','sb'],default='server',help='server retains the current BOMA-named script exactly; sb selects the saved SB-loss script.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,help='Explicit run-length override; omit to retain the archived launcher value.')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    config=json.loads((ROOT/'metadata/launch_configs.json').read_text())[a.model]
    directory=ROOT/config['cwd'];target=directory/config['scripts'][a.variant]
    output=a.output.resolve()
    if output.exists():p.error('Output already exists. Choose a fresh directory; archived results are never overwritten.')
    for d in ROOT.glob('0*'):
        if any(output.is_relative_to(d/x) for x in ['exp','data','exp_hierTFR']):p.error('Output cannot be inside an archived data or experiment directory.')
    if a.epochs is not None and a.epochs<1:p.error('Epochs must be positive.')
    arguments=list(config['arguments'])
    if a.epochs is not None:
        i=arguments.index('--n-epochs');arguments[i+1]=str(a.epochs)
    command=[sys.executable,'-B',str(target),*arguments,'--exp-dir',str(output)]
    print('Working directory:',directory)
    print('Command:',shlex.join(command))
    print('Archived semantics:',config['semantics'][a.variant])
    if a.dry_run:return
    required=[ROOT/x for x in config['required']]
    missing=[str(x.relative_to(ROOT)) for x in required if not x.is_file() or x.stat().st_size==0]
    if missing:p.error('Restore required artifacts first: '+', '.join(missing))
    # Creating the empty directory is required by the original distribution plots.
    output.mkdir(parents=True,exist_ok=False)
    record={'model':a.model,'variant':a.variant,'argv':command,'cwd':str(directory),'semantics':config['semantics'][a.variant]}
    (output/'invocation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env.setdefault('MPLBACKEND','Agg');env.setdefault('WANDB_MODE','disabled')
    with (output/'execution.log').open('xb') as log:
        code=subprocess.call(command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT)
    record['exit_code']=code
    (output/'invocation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Exit code:',code,'Log:',output/'execution.log')
    raise SystemExit(code)

if __name__=='__main__':main()
