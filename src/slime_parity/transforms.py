"""Explicit bijective parameter re-layouts for independent model references.

Only reshape, permutation, and concatenation are allowed. Every trainable
source tensor must occur exactly once, so omitted or duplicated parameters fail
without allocating model-sized index arrays. This checks storage coverage;
the semantic QKV/expert ordering still requires model-specific review/tests.
"""
from collections import Counter
from math import prod
from training_parity.common import require, shape_size


class ParameterTransform:
    def __init__(self, source_shapes, expressions):
        require(isinstance(expressions, dict) and expressions, "Declare canonical parameter expressions")
        self.sources, self.expressions = source_shapes, expressions
        self.uses = Counter()
        self.shapes = {name: self._shape(expr) for name, expr in expressions.items()}
        require(self.uses == Counter({n:1 for n in source_shapes}), "Parameter mapping must use every trainable source exactly once")

    def _shape(self, expr):
        if isinstance(expr, str):
            require(expr in self.sources, f"Unknown reference parameter: {expr}")
            self.uses[expr] += 1
            return list(self.sources[expr])
        require(isinstance(expr, dict), "Invalid parameter expression")
        op = expr.get("op")
        if op == "reshape":
            require(set(expr) == {"op", "value", "shape"}, "Invalid reshape fields")
            shape_size(expr["shape"])
            require(prod(self._shape(expr["value"])) == prod(expr["shape"]), "Reshape changes parameter elements")
            return list(expr["shape"])
        if op == "permute":
            require(set(expr) == {"op", "value", "axes"}, "Invalid permutation fields")
            shape = self._shape(expr["value"])
            axes = expr["axes"]
            require(isinstance(axes, list) and all(type(a) is int for a in axes)
                    and sorted(axes) == list(range(len(shape))), "Invalid parameter axis permutation")
            return [shape[a] for a in axes]
        if op == "concat":
            require(set(expr) == {"op", "values", "axis"} and isinstance(expr["values"], list) and expr["values"], "Invalid concat fields")
            shapes = [self._shape(x) for x in expr["values"]]
            axis = expr["axis"]
            require(type(axis) is int and 0 <= axis < len(shapes[0]), "Invalid concat axis")
            result = list(shapes[0])
            for shape in shapes[1:]:
                require(len(shape) == len(result) and all(a == b for i,(a,b) in enumerate(zip(shape,result)) if i != axis), "Incompatible concat shapes")
                result[axis] += shape[axis]
            return result
        raise ValueError(f"Unsupported parameter transform: {op}")

    def apply(self, values):
        import torch
        require(set(values) == set(self.sources), "Missing reference tensors for canonical transform")
        for name, value in values.items():
            require(list(value.shape) == list(self.sources[name]), "Source tensor shape changed")
        def evaluate(expr):
            if isinstance(expr, str):
                return values[expr]
            if expr["op"] == "reshape":
                return evaluate(expr["value"]).reshape(expr["shape"])
            if expr["op"] == "permute":
                return evaluate(expr["value"]).permute(expr["axes"])
            return torch.cat([evaluate(x) for x in expr["values"]], dim=expr["axis"])
        return {name: evaluate(expr) for name,expr in self.expressions.items()}
