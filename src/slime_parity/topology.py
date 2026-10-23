"""Exhaustive conventional Megatron dense/expert grids and runtime evidence.

This resolver implements the standard dense/expert folding convention. Native
argument validation and observed groups are additionally required at runtime.
GTP/dynamic CP/heterogeneous pipeline layouts require another named resolver;
they must not silently inherit these arithmetic rules.
"""
from dataclasses import asdict, dataclass
from itertools import product, combinations
from math import prod

from training_parity.common import object_hash, require

AXES = ("world_size", "tp", "cp", "ep", "sp", "pp", "etp", "vpp")
GROUP_AXES = ("tp", "cp", "pp", "dp", "ep", "etp", "edp")
CONVENTION = "megatron-dense-expert-v1"


@dataclass(frozen=True)
class Topology:
    world_size: int = 1
    tp: int = 1
    cp: int = 1
    ep: int = 1
    sp: bool = False
    pp: int = 1
    etp: int = 1
    vpp: int = 1

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict) and not set(value) - set(AXES), "Unknown topology field")
        t = cls(**value)
        require(type(t.sp) is bool, "SP must be a boolean")
        require(all(type(getattr(t, k)) is int and getattr(t, k) > 0
                    for k in AXES if k != "sp"), "Parallel degrees must be positive integers")
        return t

    @property
    def dp(self):
        require(self.world_size % (self.tp * self.cp * self.pp) == 0, "Nonintegral dense DP")
        return self.world_size // (self.tp * self.cp * self.pp)

    @property
    def edp(self):
        require(self.world_size % (self.etp * self.ep * self.pp) == 0, "Nonintegral expert DP")
        return self.world_size // (self.etp * self.ep * self.pp)

    def degrees(self):
        return {**{k: getattr(self, k) for k in GROUP_AXES if k not in ("dp", "edp")},
                "dp": self.dp, "edp": self.edp}

    def as_dict(self):
        return asdict(self)

    @property
    def key(self):
        return "w{}-tp{}-cp{}-ep{}-sp{}-pp{}-etp{}-vpp{}".format(
            self.world_size, self.tp, self.cp, self.ep, int(self.sp), self.pp, self.etp, self.vpp)


def resolve(topology, model, global_batch_size):
    """Return reasons without dropping requested combinations.

    Model constraints are declarations by the model adapter. Absence of optional
    shape constraints is not evidence of native support; workers must preflight.
    """
    t = topology
    errors, inapplicable = [], []
    if not model.get("num_experts", 0) and t.ep != 1:
        inapplicable.append("Dense model has no expert parallel computation")
    if t.world_size % (t.tp * t.cp * t.pp):
        errors.append("world_size is not divisible by TP*CP*PP")
    if t.world_size % (t.etp * t.ep * t.pp):
        errors.append("world_size is not divisible by ETP*EP*PP")
    if t.sp and t.tp == 1:
        errors.append("Sequence parallelism requires TP > 1 in this convention")
    if model.get("num_experts", 0) and t.tp > 1 and t.ep > 1 and not t.sp:
        errors.append("TP + EP requires sequence parallelism in this convention")
    experts = model.get("num_experts", 0)
    if experts and experts % t.ep:
        errors.append("Number of experts is not divisible by EP")
    if t.vpp > 1 and t.pp == 1:
        errors.append("Virtual pipeline requires PP > 1")
    layers = model.get("num_layers")
    if model.get("uniform_pipeline", False) and layers and layers % (t.pp * t.vpp):
        errors.append("Uniform layer layout is not divisible by PP*VPP")
    for field in ("hidden_size", "num_attention_heads", "ffn_hidden_size"):
        if field in model.get("tp_divisible", []) and model[field] % t.tp:
            errors.append(f"{field} is not divisible by TP")
    if t.world_size % (t.tp * t.cp * t.pp) == 0 and global_batch_size % t.dp:
        errors.append("Global batch is not divisible by dense DP for equal-sample scheduling")
    status = "not_applicable" if inapplicable else "invalid" if errors else "eligible"
    return {"id": t.key, "requested": t.as_dict(), "eligibility": status,
            "reasons": inapplicable + errors,
            "degrees": t.degrees() if not errors and not inapplicable else None,
            "native_preflight_required": status == "eligible"}


def expand_matrix(matrix, model, global_batch_size):
    require(isinstance(matrix, dict) and not set(matrix) - set(AXES) - {"resolver"}, "Unknown matrix field")
    require(matrix.get("resolver", CONVENTION) == CONVENTION, "Unsupported topology resolver")
    domains = {}
    for axis in AXES:
        values = matrix.get(axis, [False] if axis == "sp" else [1])
        require(isinstance(values, list) and values, f"Empty matrix dimension: {axis}")
        require(len({object_hash(v) for v in values}) == len(values), f"Duplicate {axis} values")
        for value in values:
            require(type(value) is bool if axis == "sp" else type(value) is int and value > 0,
                    f"Invalid {axis} value")
        domains[axis] = values
    require(prod(map(len, domains.values())) <= 1000000, "Matrix has over one million entries; explicitly narrow its domain")
    return [resolve(Topology.from_dict(dict(zip(AXES, values))), model, global_batch_size)
            for values in product(*(domains[a] for a in AXES))]


def validate_groups(topology, receipts):
    """Check actual memberships, not only argument values or group sizes."""
    t = topology if isinstance(topology, Topology) else Topology.from_dict(topology)
    require(len(receipts) == t.world_size, "Incomplete topology rank receipts")
    ranks = {r["rank"]: r for r in receipts}
    require(set(ranks) == set(range(t.world_size)), "Duplicate or missing topology ranks")
    degrees = t.degrees()
    for rank, row in ranks.items():
        require(row.get("topology") == t.as_dict(), "Effective topology differs from request")
        groups = row.get("groups", {})
        require(set(groups) == set(GROUP_AXES), "Missing native process groups")
        for axis, degree in degrees.items():
            group = groups[axis]
            require(isinstance(group, list) and len(group) == degree and len(set(group)) == degree
                    and rank in group and all(type(i) is int and i in ranks for i in group),
                    f"Invalid native {axis} membership")
            require(all(set(ranks[i]["groups"][axis]) == set(group) for i in group),
                    f"Nonreciprocal {axis} group")
    for axes in (("tp", "cp", "pp", "dp"), ("etp", "ep", "pp", "edp")):
        for rank, row in ranks.items():
            for a, b in combinations(axes, 2):
                require(set(row["groups"][a]) & set(row["groups"][b]) == {rank},
                        f"Overlapping native {a}/{b} axes")
                ab = {i for r in row["groups"][a] for i in ranks[r]["groups"][b]}
                ba = {i for r in row["groups"][b] for i in ranks[r]["groups"][a]}
                require(ab == ba and len(ab) == degrees[a] * degrees[b],
                        f"Native {a}/{b} axes do not commute as a Cartesian grid")
        # Pairwise intersections alone do not prove a Cartesian grid. Traverse
        # each axis once: every world rank must occur exactly once.
        for rank in ranks:
            reached = [rank]
            for axis in axes:
                reached = [i for r in reached for i in ranks[r]["groups"][axis]]
            require(len(reached) == t.world_size and set(reached) == set(ranks),
                    "Native groups do not form a complete Cartesian grid")


def observe_megatron_groups(args, models):
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.utils import get_model_config

    config = get_model_config(models[0])
    funcs = {"tp": "get_tensor_model_parallel_group", "cp": "get_context_parallel_group",
             "pp": "get_pipeline_model_parallel_group", "dp": "get_data_parallel_group",
             "ep": "get_expert_model_parallel_group", "etp": "get_expert_tensor_parallel_group",
             "edp": "get_expert_data_parallel_group"}
    groups = {axis: dist.get_process_group_ranks(getattr(ps, name)()) for axis, name in funcs.items()}
    topology = Topology(world_size=dist.get_world_size(), tp=len(groups["tp"]), cp=len(groups["cp"]),
                        pp=len(groups["pp"]), ep=len(groups["ep"]), etp=len(groups["etp"]),
                        sp=bool(config.sequence_parallel), vpp=len(models))
    require(bool(args.sequence_parallel) == topology.sp, "SP arguments and model configuration disagree")
    require(all(bool(get_model_config(m).sequence_parallel) == topology.sp for m in models), "SP differs across model chunks")
    return {"rank": dist.get_rank(), "topology": topology.as_dict(), "groups": groups}
