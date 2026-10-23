"""Test the hook's TP vocabulary / mirrored CP coordinate reconstruction."""
from types import ModuleType, SimpleNamespace
import sys

import torch

from slime_parity.adapters.slime import ForwardCapture
from slime_parity.topology import Topology


class Output(torch.nn.Module):
    output_size = 8

    def forward(self, values):
        return values, None


class Chunk(torch.nn.Module):
    def __init__(self, last):
        super().__init__()
        self.post_process = last
        self.output_layer = Output()


def test_full_tp_cp_pp_output_coordinates_include_all_tokens(tmp_path, monkeypatch):
    core = ModuleType("megatron.core")
    rank = {"tp": 0, "cp": 0}
    core.parallel_state = SimpleNamespace(get_tensor_model_parallel_rank=lambda: rank["tp"],
                                          get_context_parallel_rank=lambda: rank["cp"])
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    # Two virtual chunks on each of two PP stages; only the final chunk emits.
    t = Topology(world_size=8, tp=2, cp=2, pp=2, vpp=2)
    full = torch.arange(2*8*8, dtype=torch.float64).reshape(2, 8, 8)
    output = torch.zeros(2*6*5, dtype=torch.float64)
    cover = torch.zeros_like(output, dtype=torch.int32)
    for pp in range(2):
        for cp in range(2):
            for tp in range(2):
                rank.update(tp=tp, cp=cp)
                ctx = SimpleNamespace(suite={"description": {"observations": {"logits": {"shape": [2, 6, 5]}}}},
                                      writer=SimpleNamespace(rank_dir=tmp_path))
                current = {0: None, 1: None}
                chunks = [Chunk(False), Chunk(pp == 1)]
                hook = ForwardCapture(ctx, current, "output_layer")
                try:
                    hook.install(chunks, 8, t)
                    positions = ([0, 1, 6, 7] if cp == 0 else [2, 3, 4, 5])
                    for sample in range(2):
                        current[1] = sample
                        values = full[sample, positions, tp*4:(tp+1)*4].unsqueeze(1)
                        chunks[1].output_layer(values)
                        # A recomputation must neither double-count nor omit it.
                        chunks[1].output_layer(values.clone())
                    for interval, value in hook.fragments:
                        output[interval.start:interval.end] = value
                        cover[interval.start:interval.end] += 1
                finally:
                    hook.close()
    torch.testing.assert_close(output.reshape(2, 6, 5), full[:, :6, :5])
    assert torch.all(cover == 1)
