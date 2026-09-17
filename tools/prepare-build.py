#!/usr/bin/env python3
"""Check or apply the selected patch to a fresh source checkout, then build."""
import argparse, hashlib, json, subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=['qwen35','flash-next'])
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--build',type=Path)
    p.add_argument('--cuda-arch',help='CMake CUDA architecture for YOUR GPU; required for a build')
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--baseline',action='store_true',help='Build unmodified source for a matched control')
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]; source=a.source.resolve()
    m=json.loads((root/'patches'/f'{a.variant}-source-manifest.json').read_text())
    for f in m['files']:
        path=source/f['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=f['before_sha256']:
            p.error('Source does not match the pinned unmodified input: '+f['path'])
    patch=root/'patches'/f'{a.variant}-selected.patch'
    subprocess.run(['git','apply','--check',str(patch)],cwd=source,check=True)
    if a.check_only:
        print('Pinned affected source hashes and patch applicability passed. No files modified.');return
    if not a.build or not a.cuda_arch or a.jobs<1:p.error('A build requires --build, --cuda-arch and positive --jobs')
    build=a.build.resolve()
    if build.exists():p.error('Use a new build directory')
    if build==source or build in source.parents:p.error('Build directory cannot be the source or its parent')
    if not a.baseline:
        subprocess.run(['git','apply',str(patch)],cwd=source,check=True)
        for f in m['files']:
            assert hashlib.sha256((source/f['path']).read_bytes()).hexdigest()==f['after_sha256']
    config=['cmake','-S',str(source),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
            '-DBUILD_SHARED_LIBS=ON','-DGGML_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES='+a.cuda_arch,
            '-DGGML_NATIVE=ON','-DGGML_OPENMP=ON','-DGGML_CPU_REPACK=ON',
            '-DGGML_BACKEND_DL='+('ON' if a.variant=='flash-next' else 'OFF'),
            '-DLLAMA_BUILD_SERVER=ON','-DLLAMA_CURL=OFF']
    subprocess.run(config,check=True)
    subprocess.run(['cmake','--build',str(build),'--target','llama-server','-j',str(a.jobs)],check=True)
    print('Build complete. Validate correctness and memory headroom before ranking performance.')
if __name__=='__main__':main()
