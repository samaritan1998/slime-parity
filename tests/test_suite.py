from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import pytest
from slime_parity.adapters.mlp import example_spec
from slime_parity.suite import prepare, execute, report, load_suite
from training_parity.common import EvidenceError, read, write_new
from training_parity.plan import load_plan


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root=tmp_path_factory.mktemp("real-dp")
    spec=example_spec()
    spec['cases']=spec['cases'][:1]
    path=prepare(spec,root/'suite')
    result=execute(path)
    return path,result


def test_actual_independent_reference_and_two_rank_gloo(completed):
    path,result=completed
    assert result['status']=='passed', json.dumps(result,indent=2)
    assert result['counts']['required']==result['counts']['completed']==result['counts']['passed']==2
    row=next(r for r in result['rows'] if r['requested']['world_size']==2)
    assert len(row['comparison']['stages'][row['id']]['rank_metadata'])==2
    for step in row['comparison']['comparisons'][row['id']].values():
        assert 'exp_avg/layer0.weight' in step['tensors']
        assert 'update/layer0.weight' in step['tensors']
    assert (Path(result['report_directory'])/'report.html').is_file()


def test_resume_does_not_execute_or_rewrite_completed_runs(completed):
    path,_=completed
    logs={p:p.stat().st_mtime_ns for p in path.parent.rglob('train.log')}
    result=execute(path)
    assert result['status']=='passed'
    assert logs=={p:p.stat().st_mtime_ns for p in logs}


@pytest.mark.parametrize('fault',['gradient_sign','loss'])
def test_real_faults_are_rejected(tmp_path,fault):
    spec=example_spec();spec['cases']=spec['cases'][:1];spec['matrix']['world_size']=[1]
    spec['candidate']['options']['fault']=fault
    path=prepare(spec,tmp_path/'suite')
    result=execute(path)
    assert result['status']=='numerical_failure', json.dumps(result,indent=2)
    failures=[g for g in result['rows'][0]['comparison']['gates'] if not g['passed']]
    assert failures
    if fault=='loss':
        assert any(g['unit']=='observation/loss' for g in failures)
    else:
        assert any(g['unit']=='gradient/layer1.bias' for g in failures)


def test_resource_gap_stays_in_required_denominator(tmp_path):
    spec=example_spec();spec['cases']=spec['cases'][:1]
    spec['execution']['max_world_size']=1
    result=execute(prepare(spec,tmp_path/'suite'))
    assert result['status']=='unverified'
    assert result['counts']['required']==2 and result['counts']['passed']==1
    assert result['counts']['execution']['resource_unavailable']==1


def test_no_policy_is_report_only(tmp_path):
    spec=example_spec();spec['cases']=spec['cases'][:1];spec['matrix']['world_size']=[1];spec['policy']=[]
    result=execute(prepare(spec,tmp_path/'suite'))
    assert result['status']=='report_only'


def test_global_only_gate_cannot_qualify_model(tmp_path):
    spec=example_spec();spec['cases']=spec['cases'][:1];spec['matrix']['world_size']=[1]
    spec['policy'][0]['scope']='global'
    result=execute(prepare(spec,tmp_path/'suite'))
    assert result['status']=='report_only'
    assert 'gradient/layer0.weight' in result['rows'][0]['ungated_required_tensors']


def test_unexecuted_mixed_topologies_are_not_passed(tmp_path):
    spec=example_spec();spec['cases']=spec['cases'][:1]
    spec['matrix'].update(world_size=[8],tp=[1,2],cp=[1,2],ep=[1],sp=[False,True],pp=[1,2])
    spec['training']['global_batch_size']=8
    path=prepare(spec,tmp_path/'suite')
    result=report(path)
    assert result['status']=='unverified'
    assert result['counts']['requested']==16
    assert result['counts']['required']>1
    assert all(r['validation']=='unverified' for r in result['rows'])


def test_reference_cannot_be_same_entry(tmp_path):
    spec=example_spec();spec['reference']['runner']=spec['candidate']['runner']
    with pytest.raises(EvidenceError):prepare(spec,tmp_path/'suite')


def test_suite_tamper_rejected(tmp_path):
    spec=example_spec();spec['cases']=spec['cases'][:1]
    path=prepare(spec,tmp_path/'suite')
    suite=read(path);suite['spec']['training']['steps']=5
    path.write_text(json.dumps(suite))
    with pytest.raises(EvidenceError,match='checksum'):
        report(path)


def test_raw_corruption_invalidates_evidence(completed,tmp_path):
    # Copy only this small fixture; the original successful run remains intact.
    import shutil
    path,_=completed
    root=tmp_path/'copy';shutil.copytree(path.parent,root)
    raw=next((root/'cases/ordinary/reference').rglob('*.bin'))
    data=bytearray(raw.read_bytes());data[0]^=1;raw.write_bytes(data)
    result=report(root/'suite.json')
    assert result['status']=='unverified'
    assert result['counts']['passed']==0


def test_missing_rank_invalidates_complete_stage(completed,tmp_path):
    import shutil
    path,_=completed
    root=tmp_path/'copy';shutil.copytree(path.parent,root)
    rank=next((root/'cases/ordinary').glob('w2-*/rank1.json'))
    rank.unlink()
    result=report(root/'suite.json')
    assert result['status']=='unverified'
    assert result['counts']['validation']['incomplete_evidence']==1


def test_cli_status_for_pending_suite(tmp_path):
    spec=example_spec();spec['cases']=spec['cases'][:1]
    path=prepare(spec,tmp_path/'suite')
    run=subprocess.run([sys.executable,'-m','slime_parity','report','--suite',str(path)],capture_output=True,text=True)
    assert run.returncode==3
    assert json.loads(run.stdout)['status']=='unverified'


def test_single_gpu_slime_stage_uses_distributed_launcher(tmp_path):
    spec=example_spec();spec['cases']=spec['cases'][:1];spec['matrix']['world_size']=[1]
    spec['candidate'].update(kind='slime', runner='slime_parity.adapters.slime:run_slime', device='cuda')
    path=prepare(spec,tmp_path/'suite')
    suite=load_suite(path)
    plan=load_plan(path.parent/suite['cases'][0]['plan'])
    candidate=next(s for s in plan['stages'].values() if s['role']=='candidate')
    assert 'torch.distributed.run' in candidate['command']


def test_custom_launcher_applies_to_reference_and_all_candidate_sizes(tmp_path,monkeypatch):
    import slime_parity.suite as suite_module
    spec=example_spec();spec['cases']=spec['cases'][:1]
    spec['reference']['device']=spec['candidate']['device']='cuda'
    launcher=['cluster-submit','{world_size}','{plan}','{stage}']
    spec['execution']['launcher']=launcher
    path=prepare(spec,tmp_path/'suite')
    suite=load_suite(path)
    plan=load_plan(path.parent/suite['cases'][0]['plan'])
    assert all(s['command']==launcher for s in plan['stages'].values())
    launched=[]
    monkeypatch.setattr('torch.cuda.is_available',lambda:False)
    monkeypatch.setattr(suite_module,'run_stage',lambda path,name,**kwargs:launched.append(name))
    result=execute(path)
    assert set(launched)==set(plan['stages'])
    assert 'resource_unavailable' not in result['counts']['execution']


@pytest.mark.parametrize('domain,expected', [({'tp':[3]},'invalid'),({'ep':[2]},'not_applicable')])
def test_rejected_domain_is_reported_without_launching_reference(tmp_path,monkeypatch,domain,expected):
    import slime_parity.suite as suite_module
    spec=example_spec();spec['cases']=spec['cases'][:1];spec['matrix'].update(domain)
    path=prepare(spec,tmp_path/'suite')
    launched=[]
    monkeypatch.setattr(suite_module,'run_stage',lambda *args,**kwargs:launched.append(args))
    result=execute(path)
    assert not launched
    assert result['status']=='unverified'
    assert result['counts']['requested']==2 and result['counts']['required']==0
    assert all(row['execution']==expected and row['reasons'] for row in result['rows'])
