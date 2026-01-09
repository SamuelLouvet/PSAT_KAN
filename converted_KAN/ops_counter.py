import argparse
import importlib
import math
import operator
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from .convert import convert_to_kan


@dataclass
class OpCounts:
    adds: int = 0
    subs: int = 0
    muls: int = 0
    divs: int = 0
    pow: int = 0
    exp: int = 0
    relu: int = 0
    reciprocal: int = 0
    sqrt: int = 0
    max: int = 0
    compare: int = 0

    def total_arith(self) -> int:
        return (
            self.adds
            + self.subs
            + self.muls
            + self.divs
            + self.pow
            + self.exp
            + self.relu
            + self.reciprocal
            + self.sqrt
            + self.max
            + self.compare
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "adds": self.adds,
            "subs": self.subs,
            "muls": self.muls,
            "divs": self.divs,
            "pow": self.pow,
            "exp": self.exp,
            "relu": self.relu,
            "reciprocal": self.reciprocal,
            "sqrt": self.sqrt,
            "max": self.max,
            "compare": self.compare,
            "total_arith": self.total_arith(),
        }


def _numel(shape: Iterable[int]) -> int:
    return int(reduce(operator.mul, shape, 1))


def _tensor_numel(meta: Any) -> int:
    if meta is None:
        return 0
    shape = getattr(meta, "shape", None)
    if shape is None:
        return 0
    return _numel(shape)


def _get_meta(node: fx.Node) -> Any:
    return node.meta.get("tensor_meta")


def _as_int_tuple(value: Any) -> Tuple[int, ...]:
    return tuple(int(v) for v in value)


def _matmul_counts(a_shape: Tuple[int, ...], b_shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(a_shape) < 2 or len(b_shape) < 2:
        return 0, 0

    *batch_a, m, k1 = a_shape
    *batch_b, k2, n = b_shape
    if k1 != k2:
        return 0, 0

    batch = 1
    for b in batch_a:
        batch *= b
    if batch_b:
        for b in batch_b:
            batch *= b

    muls = batch * m * n * k1
    adds = batch * m * n * max(k1 - 1, 0)
    return adds, muls


def _sum_reduction_counts(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> int:
    in_numel = _numel(input_shape)
    out_numel = _numel(output_shape)
    if out_numel == 0:
        return 0
    reduce_size = in_numel // out_numel
    if reduce_size <= 1:
        return 0
    return out_numel * (reduce_size - 1)


def _to_2tuple(x: Any) -> Tuple[int, int]:
    if isinstance(x, tuple):
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))


def _load_model(spec: str) -> torch.nn.Module:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError("Model spec must be like 'module:attr'")

    mod = importlib.import_module(module_name)
    obj = getattr(mod, attr)
    model = obj() if callable(obj) else obj
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Loaded object is not an nn.Module")
    return model


def count_ops(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    per_layer: bool = False,
) -> Dict[str, Any]:
    model = model.to(device=device, dtype=dtype).eval()

    example_input = torch.zeros(input_shape, device=device, dtype=dtype)
    gm = fx.symbolic_trace(model)
    ShapeProp(gm).propagate(example_input)

    counts = OpCounts()
    per_layer_counts: Dict[str, OpCounts] = {}

    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method", "call_module"):
            continue

        tgt = node.target
        meta = _get_meta(node)
        out_numel = _tensor_numel(meta)

        layer_name = None
        if per_layer:
            stack = node.meta.get("nn_module_stack")
            if stack:
                # Take the last module in the stack (closest owner).
                layer_name = list(stack.values())[-1][0]

        def _accumulate(field: str, value: int) -> None:
            if value == 0:
                return
            setattr(counts, field, getattr(counts, field) + value)
            if per_layer and layer_name is not None:
                layer = per_layer_counts.setdefault(layer_name, OpCounts())
                setattr(layer, field, getattr(layer, field) + value)

        if node.op in ("call_function", "call_method"):
            if tgt in (operator.add, torch.add, torch.Tensor.add, "add"):
                _accumulate("adds", out_numel)
            elif tgt in (operator.sub, torch.sub, torch.Tensor.sub, "sub"):
                _accumulate("subs", out_numel)
            elif tgt in (operator.mul, torch.mul, torch.Tensor.mul, "mul"):
                _accumulate("muls", out_numel)
            elif tgt in (operator.truediv, torch.div, torch.Tensor.div, "div"):
                _accumulate("divs", out_numel)
            elif tgt in (torch.exp, torch.Tensor.exp, "exp"):
                _accumulate("exp", out_numel)
            elif tgt in (torch.relu, torch.nn.functional.relu, torch.Tensor.relu, "relu"):
                _accumulate("relu", out_numel)
            elif tgt in (torch.reciprocal, torch.Tensor.reciprocal, "reciprocal"):
                _accumulate("reciprocal", out_numel)
            elif tgt in (torch.sqrt, torch.Tensor.sqrt, "sqrt"):
                _accumulate("sqrt", out_numel)
            elif tgt in (torch.maximum, torch.max, "maximum", "max"):
                _accumulate("max", out_numel)
            elif tgt in (operator.pow, torch.pow, torch.Tensor.pow, "pow"):
                _accumulate("pow", out_numel)
            elif tgt in (torch.sum, torch.Tensor.sum, "sum"):
                inp = node.args[0]
                in_meta = _get_meta(inp) if isinstance(inp, fx.Node) else None
                in_shape = _as_int_tuple(in_meta.shape) if in_meta is not None else ()
                out_shape = _as_int_tuple(meta.shape) if meta is not None else ()
                _accumulate("adds", _sum_reduction_counts(in_shape, out_shape))
            elif tgt in (torch.mean, torch.Tensor.mean, "mean"):
                inp = node.args[0]
                in_meta = _get_meta(inp) if isinstance(inp, fx.Node) else None
                in_shape = _as_int_tuple(in_meta.shape) if in_meta is not None else ()
                out_shape = _as_int_tuple(meta.shape) if meta is not None else ()
                _accumulate("adds", _sum_reduction_counts(in_shape, out_shape))
                _accumulate("divs", _numel(out_shape))
            elif tgt in (torch.matmul, torch.mm, torch.bmm, "matmul", "mm", "bmm"):
                a = node.args[0]
                b = node.args[1]
                a_meta = _get_meta(a) if isinstance(a, fx.Node) else None
                b_meta = _get_meta(b) if isinstance(b, fx.Node) else None
                if a_meta is not None and b_meta is not None:
                    adds, muls = _matmul_counts(
                        _as_int_tuple(a_meta.shape), _as_int_tuple(b_meta.shape)
                    )
                    _accumulate("adds", adds)
                    _accumulate("muls", muls)
            continue

        if node.op == "call_module":
            module = gm.get_submodule(node.target)
            if isinstance(module, nn.ReLU):
                _accumulate("relu", out_numel)
            elif isinstance(module, nn.Linear):
                inp = node.args[0]
                in_meta = _get_meta(inp) if isinstance(inp, fx.Node) else None
                in_shape = _as_int_tuple(in_meta.shape) if in_meta is not None else ()
                if len(in_shape) >= 2:
                    in_features = module.in_features
                    muls = out_numel * in_features
                    adds = out_numel * max(in_features - 1, 0)
                    _accumulate("muls", muls)
                    _accumulate("adds", adds)
                    if module.bias is not None:
                        _accumulate("adds", out_numel)
            elif isinstance(module, nn.Conv2d):
                kH, kW = _to_2tuple(module.kernel_size)
                cin_g = module.in_channels // module.groups
                kernel_mul = cin_g * kH * kW
                muls = out_numel * kernel_mul
                adds = out_numel * max(kernel_mul - 1, 0)
                _accumulate("muls", muls)
                _accumulate("adds", adds)
                if module.bias is not None:
                    _accumulate("adds", out_numel)
            elif isinstance(module, nn.AvgPool2d):
                kH, kW = _to_2tuple(module.kernel_size)
                kernel = kH * kW
                _accumulate("adds", out_numel * max(kernel - 1, 0))
                _accumulate("divs", out_numel)
            elif isinstance(module, nn.MaxPool2d):
                kH, kW = _to_2tuple(module.kernel_size)
                kernel = kH * kW
                _accumulate("compare", out_numel * max(kernel - 1, 0))
            elif isinstance(module, nn.Softmax):
                inp = node.args[0]
                in_meta = _get_meta(inp) if isinstance(inp, fx.Node) else None
                in_shape = _as_int_tuple(in_meta.shape) if in_meta is not None else ()
                _accumulate("exp", out_numel)
                if in_shape:
                    _accumulate("adds", _sum_reduction_counts(in_shape, _as_int_tuple(meta.shape)))
                _accumulate("divs", out_numel)
            continue


    result: Dict[str, Any] = {"total": counts.as_dict()}
    if per_layer:
        result["per_layer"] = {
            name: layer_counts.as_dict() for name, layer_counts in per_layer_counts.items()
        }
    return result


def _parse_shape(text: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("input shape must be a comma-separated list of ints")
    return tuple(int(p) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count arithmetic ops for a model.")
    parser.add_argument("--model", required=True, help="Model spec: module:attr")
    parser.add_argument("--input", required=True, help="Input shape, e.g. 1,3,32,32")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--dtype", default="float32", help="float32 or float64")
    parser.add_argument(
        "--convert-kan",
        action="store_true",
        help="Convert supported layers to KAN before counting",
    )
    parser.add_argument(
        "--compare-kan",
        action="store_true",
        help="Print both non-KAN and KAN counts",
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="Include per-layer counts in output",
    )
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    model = _load_model(args.model)
    input_shape = _parse_shape(args.input)

    def _print_counts(header: str, result: Dict[str, Any]) -> None:
        print(header)
        for k, v in result["total"].items():
            print(f"{k}: {v}")
        if args.per_layer:
            print("\nPer-layer:")
            for name, layer_counts in result["per_layer"].items():
                print(f"{name}: {layer_counts}")
        print("")

    if args.compare_kan:
        base_counts = count_ops(
            model,
            input_shape,
            device=args.device,
            dtype=dtype,
            per_layer=args.per_layer,
        )
        kan_model = convert_to_kan(model, inplace=False)
        kan_counts = count_ops(
            kan_model,
            input_shape,
            device=args.device,
            dtype=dtype,
            per_layer=args.per_layer,
        )
        _print_counts("Non-KAN:", base_counts)
        _print_counts("KAN:", kan_counts)
        return

    if args.convert_kan:
        model = convert_to_kan(model, inplace=False)

    counts = count_ops(
        model,
        input_shape,
        device=args.device,
        dtype=dtype,
        per_layer=args.per_layer,
    )
    for k, v in counts["total"].items():
        print(f"{k}: {v}")
    if args.per_layer:
        print("\nPer-layer:")
        for name, layer_counts in counts["per_layer"].items():
            print(f"{name}: {layer_counts}")


if __name__ == "__main__":
    main()
