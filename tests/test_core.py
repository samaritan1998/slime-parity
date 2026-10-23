import subprocess
import sys
import numpy as np
import pytest

from training_parity import prepare, RankWriter, finalize_stage, compare, EvidenceError
from training_parity.common import read, object_hash
from training_parity.plan import load_plan
from training_parity.runner import run_stage


def small_spec():
    return {'schema_version':1,'name':'shards','contract':{'input':'fixed','initial_state':'fixed','objective':'test'},
            'steps':[0,1], 'tensors':{'gradient/w':{'shape':[6],'role':'gradient'}},
            'stages':{'reference':{'role':'reference'},'candidate':{'role':'candidate','world_size':2}},
            'policy':[{'id':'local','scope':'tensor','limits':{'atol':1e-12,'rtol':1e-10}}]}


def test_different_shards_and_fp64_preservation(tmp_path):
    path=prepare(small_spec(),tmp_path/'run');plan=load_plan(path)
    for stage in ['reference','candidate']:
        for rank in range(plan['stages'][stage]['world_size']):
            with RankWriter(path,stage,rank) as writer:
                start,end=(0,6) if stage=='reference' else ((0,2) if rank==0 else (2,6))
                for step in plan['steps']:
                    values=np.arange(start,end,dtype=np.float64)+step
                    if stage=='candidate' and rank==1:values[-1]+=1e-7
                    writer.write(step,'gradient/w',values,start=start)
                writer.finish(observed_contract=plan['contract'],configuration=plan['stages'][stage]['configuration'],completed_steps=plan['steps'])
        finalize_stage(path,stage,returncode=0)
    result=compare(path)
    assert result['status']=='numerical_failure'
    assert result['comparisons']['candidate']['0']['tensors']['gradient/w']['max_abs_location']['flat_index']==5


def test_nonfinite_and_duplicate_json_rejected(tmp_path):
    p=tmp_path/'bad.json'
    for value in ['{"x":1,"x":2}','{"x":1e999}','{"x":NaN}']:
        p.write_text(value)
        with pytest.raises(EvidenceError):read(p)


def test_runner_timeout_preserves_failure(tmp_path):
    spec=small_spec()
    spec['stages']['reference']['command']=['{python}','-c','import time;time.sleep(60)','{plan}','{stage}']
    path=prepare(spec,tmp_path/'run')
    with pytest.raises(subprocess.TimeoutExpired):run_stage(path,'reference',timeout=0.05)
    assert (path.parent/'reference/failure.json').is_file()
    assert not (path.parent/'reference/complete.json').exists()


def test_fp64_squared_underflow_cannot_manufacture_a_pass(tmp_path):
    spec=small_spec()
    spec['tensors']['gradient/w']['shape']=[1]
    spec['stages']['candidate']['world_size']=1
    spec['policy'][0]['limits']={'relative_l2':1e-6,'cosine_min':.999}
    path=prepare(spec,tmp_path/'tiny');plan=load_plan(path)
    for stage,value in [('reference',1e-200),('candidate',2e-200)]:
        with RankWriter(path,stage,0) as writer:
            for step in plan['steps']:
                writer.write(step,'gradient/w',np.array([value],dtype=np.float64))
            writer.finish(observed_contract=plan['contract'],configuration=plan['stages'][stage]['configuration'],completed_steps=plan['steps'])
        finalize_stage(path,stage,returncode=0)
    result=compare(path)
    assert result['status']=='numerical_failure'
    assert result['comparisons']['candidate']['0']['tensors']['gradient/w']['relative_l2']==1
