"""FP64 streaming statistics; zero reference and nonfinite values stay explicit."""
import math
import numpy as np
from .common import require


def floating_array(value):
    array = np.asarray(value)
    require(array.dtype.kind == "f" and array.dtype.itemsize <= 8,
            "Expected real floating tensors of at most 64 bits")
    require(array.size > 0 and np.isfinite(array).all(), "Empty or nonfinite tensor")
    return array


class Accumulator:
    def __init__(self):
        self.elements = 0
        self.scales = {key: 0.0 for key in ("reference", "candidate", "difference")}
        self.sumsq = dict(self.scales)
        self.dot = self.max_abs = 0.0

    def add(self, reference, candidate):
        left = floating_array(reference).astype(np.float64, copy=False).reshape(-1)
        right = floating_array(candidate).astype(np.float64, copy=False).reshape(-1)
        require(left.shape == right.shape, "Metric shapes differ")
        with np.errstate(over="ignore", invalid="ignore"):
            diff = right - left
        require(np.isfinite(diff).all(), "FP64 difference overflow")
        part = Accumulator()
        part.elements = left.size
        normalized = {}
        # LAPACK-style scaled sums of squares. Squaring the raw FP64 values
        # would underflow for small gradients and could manufacture a pass.
        for key, values in (("reference", left), ("candidate", right), ("difference", diff)):
            scale = float(np.max(np.abs(values)))
            unit = values / scale if scale else values
            part.scales[key] = scale
            part.sumsq[key] = float(np.dot(unit, unit))
            normalized[key] = unit
        part.dot = float(np.dot(normalized["reference"], normalized["candidate"]))
        part.max_abs = part.scales["difference"]
        return self.merge(part)

    def merge(self, other):
        scales = {key: max(self.scales[key], other.scales[key]) for key in self.scales}
        left = {key: self.scales[key] / scale if scale else 0.0 for key,scale in scales.items()}
        right = {key: other.scales[key] / scale if scale else 0.0 for key,scale in scales.items()}
        self.dot = (self.dot * left["reference"] * left["candidate"]
                    + other.dot * right["reference"] * right["candidate"])
        for key in scales:
            self.sumsq[key] = self.sumsq[key] * left[key] ** 2 + other.sumsq[key] * right[key] ** 2
        self.scales = scales
        self.elements += other.elements
        self.max_abs = max(self.max_abs, other.max_abs)
        return self

    def result(self):
        require(self.elements > 0, "No elements compared")
        require(all(math.isfinite(v) for v in [*self.scales.values(), *self.sumsq.values(), self.dot, self.max_abs]),
                "FP64 aggregate overflow")
        roots = {key: math.sqrt(value) for key,value in self.sumsq.items()}
        ref, actual, delta = (self.scales[key] * roots[key] for key in ("reference", "candidate", "difference"))
        require(all(math.isfinite(v) for v in (ref, actual, delta)), "FP64 norm overflow")
        cosine = (self.dot / roots["reference"] / roots["candidate"]
                  if ref and actual else (1.0 if not delta else 0.0))
        if ref:
            # Do not divide the rounded absolute subnormal norms. Keep their
            # scales separate until the final ratio, including extreme exponents.
            dm, de = math.frexp(self.scales["difference"])
            rm, re = math.frexp(self.scales["reference"])
            try:
                relative = math.ldexp((dm / rm) * (roots["difference"] / roots["reference"]), de - re)
            except OverflowError:
                relative = math.inf
        else:
            relative = 0.0 if not delta else None
        require(relative is None or math.isfinite(relative), "FP64 relative error overflow")
        if ref and delta and relative == 0:
            relative = None  # Unrepresentable nonzero ratio cannot satisfy a zero-error gate.
        rmse = self.scales["difference"] * (roots["difference"] / math.sqrt(self.elements))
        if delta and rmse == 0:
            rmse = None
        return {"elements": int(self.elements), "max_abs": self.max_abs,
                "absolute_l2": delta, "rmse": rmse,
                "reference_l2": ref, "candidate_l2": actual,
                "relative_l2": relative,
                "cosine": max(-1.0, min(1.0, cosine))}


def compare_arrays(reference, candidate):
    """Return metrics only, with no implicit tolerance or acceptance verdict."""
    left, right = floating_array(reference), floating_array(candidate)
    require(left.shape == right.shape, "Tensor shapes differ")
    return Accumulator().add(left, right).result()
