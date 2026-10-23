from copy import deepcopy
from itertools import product
import pytest

from slime_parity.topology import Topology, expand_matrix, validate_groups, GROUP_AXES
from training_parity.common import EvidenceError


def dp_receipts(world=2):
    t = Topology(world_size=world)
    return [{"rank": rank, "topology": t.as_dict(), "groups": {
        a: list(range(world)) if a in ("dp", "edp") else [rank] for a in GROUP_AXES}}
        for rank in range(world)]


def test_all_combinations_retained_and_expert_folding():
    domain = {"world_size": [8,16], "tp": [1,2], "cp": [1,2], "ep": [1,2],
              "sp": [False,True], "pp": [1,2], "etp": [1,2], "vpp": [1,2]}
    rows = expand_matrix(domain, {"num_experts": 4, "num_layers": 8, "uniform_pipeline": True}, 32)
    assert len(rows) == 256
    assert len({r["id"] for r in rows}) == 256
    mixed = next(r for r in rows if r["requested"] == dict(world_size=8,tp=2,cp=2,ep=2,sp=True,pp=2,etp=1,vpp=1))
    assert mixed["eligibility"] == "eligible"
    assert mixed["degrees"]["dp"] == 1
    assert mixed["degrees"]["edp"] == 2
    assert any(r["eligibility"] == "invalid" for r in rows)


@pytest.mark.parametrize("field,value", [("tp",True),("sp",1),("cp",0),("pp",-1),("world_size",1.5)])
def test_reject_ambiguous_values(field,value):
    with pytest.raises(EvidenceError):
        expand_matrix({field:[value]}, {}, 4)


def test_dense_experts_not_silently_removed():
    rows = expand_matrix({"world_size":[4],"ep":[1,2]}, {"num_experts":0}, 4)
    assert len(rows) == 2
    assert rows[1]["eligibility"] == "not_applicable"


def test_native_group_validation():
    rows = dp_receipts()
    validate_groups(Topology(world_size=2), rows)
    bad = deepcopy(rows)
    bad[1]["groups"]["dp"] = [1]
    with pytest.raises(EvidenceError):
        validate_groups(Topology(world_size=2), bad)
    with pytest.raises(EvidenceError):
        validate_groups(Topology(world_size=2), [rows[0],rows[0]])


def test_native_sp_not_just_flags():
    rows = dp_receipts(1)
    rows[0]["topology"]["sp"] = True
    with pytest.raises(EvidenceError, match="Effective topology"):
        validate_groups(Topology(), rows)


def test_invalid_microbatch_contributions_and_vpp():
    rows = expand_matrix({"world_size":[8],"vpp":[1,2]}, {}, 3)
    assert all(r["eligibility"] == "invalid" for r in rows)
    assert any("Global batch" in reason for reason in rows[0]["reasons"])


def test_unknown_topology_convention_rejected():
    with pytest.raises(EvidenceError):
        expand_matrix({"resolver":"gtp-unimplemented"}, {}, 4)


def test_duplicate_domains_rejected():
    with pytest.raises(EvidenceError):
        expand_matrix({"tp":[1,1]}, {}, 4)


def test_noncommuting_axes_cannot_claim_a_cartesian_topology():
    t = Topology(world_size=8, tp=2, cp=2, pp=2)
    rows=[]
    for rank in range(8):
        i,j=divmod(rank,2)
        groups={'tp':sorted([rank,2*i+1-j]),'cp':sorted([rank,2*((i+2)%4)+j]),
                'pp':sorted([rank,2*((i+(1 if j==0 else -1))%4)+1-j]),
                'dp':[rank],'etp':[rank],'ep':[rank],
                'edp':[r for r in range(8) if (r//2)%2==i%2]}
        rows.append({'rank':rank,'topology':t.as_dict(),'groups':groups})
    with pytest.raises(EvidenceError,match='do not commute'):
        validate_groups(t,rows)


def test_folded_dense_expert_cartesian_groups_are_accepted():
    t=Topology(world_size=16,tp=2,cp=2,pp=2,ep=2,etp=1,sp=True)
    # Distinct dense and expert grids share the same PP partition.
    dense={r:(r%2,(r//2)%2,r//8,(r//4)%2) for r in range(16)}
    expert={r:(0,r%2,r//8,(r%8)//2) for r in range(16)}
    rows=[]
    for rank in range(16):
        groups={}
        for grid,axes in ((dense,('tp','cp','pp','dp')),(expert,('etp','ep','pp','edp'))):
            for axis, name in enumerate(axes):
                groups[name]=[other for other in grid if all(
                    grid[rank][k]==grid[other][k] for k in range(4) if k!=axis)]
        rows.append({'rank':rank,'topology':t.as_dict(),'groups':groups})
    validate_groups(t,rows)
