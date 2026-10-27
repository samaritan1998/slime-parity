"""Observe real standard PyTorch optimizer steps for independent references."""
from training_parity.common import require
from .optimizer_counter import check_counter


class TorchStep:
    def __init__(self, context, step, parameters, optimizer):
        import torch
        require(type(optimizer) in (torch.optim.Adam, torch.optim.AdamW), "Reference observer supports native Adam/AdamW")
        self.ctx, self.step, self.params, self.optimizer = context, step, parameters, optimizer
        expected = context.suite["description"]["parameters"]
        require(set(parameters) == set(expected), "Reference trainable parameter set differs")
        require(len({id(p) for p in parameters.values()}) == len(parameters), "Canonical aliases must have one owner")
        require({id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in parameters.values()},
                "Optimizer parameter set differs from model")
        require(all(list(p.shape) == expected[n]["shape"] for n,p in parameters.items()), "Canonical parameter shapes differ")
        require(all(p.dtype in (torch.float32, torch.float64) for p in parameters.values()),
                "Standard torch observer requires FP32/FP64 optimizer parameters; use a master-weight adapter for low precision")
        self.before = {n: p.detach().cpu().clone() for n,p in parameters.items()}
        for name, value in self.before.items():
            context.write(step, "initial", name, value)
        self.captured = False

    def capture_gradients(self):
        require(not self.captured, "Gradients already captured")
        for name, p in self.params.items():
            require(p.grad is not None, f"Missing gradient (not implicitly zero-filled): {name}")
            self.ctx.write(self.step, "gradient", name, p.grad)
        self.captured = True

    def capture_update(self):
        require(self.captured, "Capture pre-clip gradients first")
        for name, p in self.params.items():
            state = self.optimizer.state[p]
            require("step" in state, "Missing optimizer counter")
            check_counter(state["step"], self.step + 1, "Optimizer")
            self.ctx.write(self.step, "parameter", name, p)
            self.ctx.write(self.step, "update", name, p.detach().cpu() - self.before[name])
            for field in ("exp_avg", "exp_avg_sq"):
                require(field in state, f"Missing native optimizer state: {field}")
                self.ctx.write(self.step, field, name, state[field])
        self.before.clear()
