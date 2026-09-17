#!/usr/bin/env python3
"""Launch an explicitly selected runtime with a portable experiment profile."""
import argparse,json,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--server',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--projector',type=Path)
    p.add_argument('--port',type=int,default=18081)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not 1024<=a.port<=65535:p.error('Choose a port from 1024 through 65535')
    d=json.loads(a.profile.read_text())
    if '{PROJECTOR}' in d['args'] and not a.projector:p.error('This profile requires --projector to preserve the measured memory conditions')
    for path in [a.server,a.model]+([a.projector] if a.projector else []):
        if not path.is_file():p.error('Required file missing: '+str(path))
    sub={'{MODEL}':str(a.model.resolve()),'{PROJECTOR}':str(a.projector.resolve()) if a.projector else '', '{PORT}':str(a.port)}
    args=[sub.get(x,x) for x in d['args']]
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_'))}
    env.update(d['environment'])
    server=a.server.resolve()
    env['LD_LIBRARY_PATH']=str(server.parent)+os.pathsep+str(server.parent.parent/'lib')+(os.pathsep+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    if a.dry_run:
        print(json.dumps({'args':args,'experiment_environment':d['environment']},indent=2));return
    os.execve(str(server),[str(server),*args],env)
if __name__=='__main__':main()
