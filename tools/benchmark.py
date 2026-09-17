#!/usr/bin/env python3
"""Synthetic public smoke workloads. These do not reproduce historical scores."""
import argparse,json,time,hashlib,urllib.request,urllib.parse
from pathlib import Path

def fixtures():
    code=''.join(f'int value_{i}(void) {{ return {i}; }}\n' for i in range(80))
    records='\n'.join(f'item_{i:04d}={i*17+3}' for i in range(1200))
    return {
      'edit':('Return the following C source verbatim, except change the return value of value_37 from 37 to 999. Output only the source, without fences, preserving the final newline.\n'+code,code.replace('value_37(void) { return 37; }','value_37(void) { return 999; }'),4096),
      'extract':('Read the records and return only the integer value of item_1097.\n'+records,str(1097*17+3),64),
      'prose':('Explain why larger prompt batches can improve mixture-of-experts prefill while making autoregressive decoding slower. Use about 200 words.',None,512)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:18081')
    p.add_argument('--model',default='benchmark-model')
    p.add_argument('--workload',choices=list(fixtures()),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=1800)
    a=p.parse_args();url=urllib.parse.urlparse(a.url)
    if url.hostname not in ('localhost','127.0.0.1','::1') or url.scheme!='http':p.error('Only local HTTP benchmark endpoints are accepted')
    if a.output.exists():p.error('Use a fresh output filename')
    prompt,expected,cap=fixtures()[a.workload]
    payload={'model':a.model,'messages':[{'role':'user','content':prompt}], 'temperature':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'seed':777,'max_tokens':cap,'cache_prompt':False,'stream':False}
    req=urllib.request.Request(a.url.rstrip('/')+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    begin=time.monotonic()
    with urllib.request.urlopen(req,timeout=a.timeout) as response: result=json.load(response)
    elapsed=time.monotonic()-begin
    choice=result['choices'][0];text=choice['message'].get('content') or ''
    exact=text==expected if expected is not None else None
    check=(text.strip()==expected) if a.workload=='extract' else exact
    record={'scope':'NEW synthetic workload, not a replay of historical private inputs','workload':a.workload,
        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'wall_seconds':elapsed,
        'finish_reason':choice.get('finish_reason'),'exact_bytes_match':exact,'task_check':check,
        'answer':text,'quality_note':'Prose requires manual review; no automatic passing score.' if expected is None else 'Edit checks exact bytes including final newline; extraction permits surrounding whitespace.',
        'usage':{k:v for k,v in result.get('usage',{}).items() if isinstance(v,(int,float))}}
    # Deliberately exclude server-reported paths, metadata and arbitrary response fields.
    with a.output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps({k:record[k] for k in ['workload','wall_seconds','task_check','finish_reason']}))
if __name__=='__main__':main()
