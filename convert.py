import builtins
import inspect
import pickle
from collections import OrderedDict
from typing import Callable

import megengine.functional
import megengine.module
import torch
import torch.fx
import torch.nn.modules as M
from torch.fx import symbolic_trace


class ModuleConverter:

    mge_cls = None
    kwargs_mapping = {}

    def __init__(self, module: M):
        self.torch_module = module

    def get_mge_cls(self) -> megengine.module.Module:
        if self.mge_cls:
            return self.mge_cls
        cls_name = type(self.torch_module).__name__
        mge_cls = getattr(megengine.module, cls_name)
        return mge_cls

    def get_args(self):
        parameters = inspect.signature(
            getattr(self.get_mge_cls(), "__init__")
        ).parameters
        params = OrderedDict()
        for mk, _v in parameters.items():
            if mk in {"self", "kwargs", "args", "name"}:
                continue
            tk = mk
            if mk in self.kwargs_mapping:
                tk = self.kwargs_mapping[mk]
                if isinstance(tk, Callable):
                    params[mk] = tk(self.torch_module)
                    continue
                if tk is None:
                    continue
            assert hasattr(
                self.torch_module, tk
            ), f"{self.torch_module} has no attr `{tk}'"
            params[mk] = getattr(self.torch_module, tk)
        return params

    def convert(self):
        cls_name = self.get_mge_cls().__name__
        return f"M.{cls_name}({self._format_args(self.get_args())})"

    def _format_args(self, kwargs):
        kwargs_s = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
        return kwargs_s


_module_converters = {}


def _register_module(*module):
    def cvt(impl):
        for m in module:
            _module_converters[m] = impl
        return impl

    return cvt


@_register_module(M.Conv2d, M.Linear)
class Convert(ModuleConverter):

    kwargs_mapping = {
        "bias": lambda m: m.bias is not None,
        "conv_mode": None,
        "compute_mode": None,
    }


@_register_module(M.BatchNorm2d)
class Convert(ModuleConverter):

    kwargs_mapping = {"freeze": None}


@_register_module(M.InstanceNorm2d)
class Convert(ModuleConverter):

    mge_cls = megengine.module.InstanceNorm
    kwargs_mapping = {"num_channels": "num_features"}


@_register_module(M.AdaptiveAvgPool2d)
class Convert(ModuleConverter):

    kwargs_mapping = {"oshp": "output_size"}


def module_converter_factory(module: M):
    if type(module) in _module_converters:
        return _module_converters[type(module)](module)
    return ModuleConverter(module)


class FunctionConvert:

    mge_fn = None
    kwargs_mapping = {}

    def __init__(self, node):
        self.node = node
        self.torch_fn = node.target

    def get_args(self, args, kwargs):
        kwargs = dict(kwargs)
        keys = list(kwargs.keys())
        for k in keys:
            if k in self.kwargs_mapping:
                mge_k = self.kwargs_mapping[k]
                if mge_k is None:
                    kwargs.pop(k)
                    continue
                else:
                    kwargs[mge_k] = kwargs.pop(k)
        return _format_args(args, kwargs)

    def convert(self, args, kwargs):
        target = self.torch_fn
        fn = self.mge_fn or self.torch_fn.__name__
        args_s = self.get_args(args, kwargs)
        if hasattr(megengine.functional, fn):
            return f"F.{fn}({args_s})"
        elif target.__module__ == "_operator":
            return f"operator.{fn}({args_s})"
        elif hasattr(builtins, fn):
            return f"{fn}({args_s})"
        else:
            raise RuntimeError(f"not support {target.__module__} {target.__name__}")


_function_converters = {}


def _register_function(*function):
    def cvt(impl):
        for m in function:
            _function_converters[m] = impl
        return impl

    return cvt


@_register_function(torch.cat)
class Convert(FunctionConvert):
    mge_fn = "concat"
    kwargs_mapping = {"dim": "axis"}


@_register_function(torch.split)
class Convert(FunctionConvert):
    def convert(self, args, kwargs):
        x = args[0]
        sp = args[1]
        if isinstance(sp, int):
            return (
                f"F.split({x}, {x}.shape[{kwargs['dim']}]//{sp}, axis={kwargs['dim']})"
            )
        else:
            return f"F.split({self.get_args(args, kwargs)})"


@_register_function(torch.transpose)
class Convert(FunctionConvert):
    def convert(self, args, kwargs):
        return f"F.transpose({args[0]}, Helper.transpose_pat({args[0]}.ndim, {args[1]}, {args[2]}))"


def function_converter_factory(func):
    if func in _function_converters:
        return _function_converters[func]
    return FunctionConvert


def _get_target(path):
    if len(path) == 1:
        return f"{path[0]}"
    r = path[0]
    for e in path[1:]:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f"{r}.{e}"
    return r


def _set_target(path, val):
    if len(path) == 1:
        return f"{path[0]} = {val}"
    r = _get_target(path[:-1])
    e = path[-1]
    if not e.isidentifier():
        r = f'setattr({r}, "{e}", {val})'
    else:
        r = f"{r}.{e} = {val}"
    return r


def _format_args(args, kwargs):
    args_s = ", ".join(repr(a) for a in args)
    kwargs_s = ", ".join(f"{k} = {repr(v)}" for k, v in kwargs.items())
    if args_s and kwargs_s:
        return f"{args_s}, {kwargs_s}"
    return args_s or kwargs_s


class VisitorContext:
    parent_module = []
    code = []


def visit_module(name, module, ctx: VisitorContext):
    if len(list(module.children())) > 0:
        ctx.code.append(_set_target(ctx.parent_module + [name], "Module()"))
    else:
        code = module_converter_factory(module).convert()
        ctx.code.append(_set_target(ctx.parent_module + [name], code))
    ctx.parent_module.append(name)
    for name, child in module.named_children():
        visit_module(name, child, ctx)
    ctx.parent_module.pop(-1)


def convert_code(symbolic_traced):
    code = []
    for node in list(symbolic_traced.graph.nodes):
        if node.op == "call_module":
            target = ["root"] + node.target.split(".")
            code.append(
                f"{node.name}={_get_target(target)}({_format_args(node.args, node.kwargs)})"
            )
        elif node.op == "call_function":
            cvt = function_converter_factory(node.target)(node)
            code.append(f"{node.name}={cvt.convert(node.args, node.kwargs)}")
        elif node.op == "call_method":
            target = node.target
            if target == "contiguous":
                code.append(f"{node.name}={node.args[0]}")
            elif target == "view":
                code.append(f"{node.name}={node.args[0]}.reshape{node.args[1:]}")
            elif target == "size":
                if len(node.args) > 1:
                    code.append(f"{node.name}={node.args[0]}.shape[{node.args[1]}]")
                else:
                    code.append(f"{node.name}={node.args[0]}.shape")
            elif target == "chunk":
                code.append(
                    f"{node.name}=F.split({node.args[0]}, {node.args[1]}, axis={node.kwargs['dim']})"
                )
            else:
                code.append(f"{node.name}={node.args[0]}.{target}{node.args[1:]}")
        elif node.op == "output":
            code.append(f"return {node.args[0]}")

    return code


template = """import operator
import pickle
import numpy as np
import megengine.module as M
import megengine.functional as F
from megengine import jit, tensor


class Module(M.Module):

    def forward(self, inputs):
        return super().forward(inputs)

class Helper:

    @staticmethod
    def transpose_pat(ndim, a, b):
        pat = list(range(ndim))
        pat[a], pat[b] = pat[b], pat[a]
        return pat
    

{module_code}

@jit.trace(capture_as_const=True)
def forward(x):
{forward_code}

with open("state.pkl", "rb") as f:
    state = pickle.load(f)
tstate = root.state_dict()
for k in tstate.keys():
    state[k] = state[k].reshape(tstate[k].shape)
root.load_state_dict(state, strict=False)
data = tensor(np.random.random([1, 3, 224, 224]).astype(np.float32))

root.eval()
ret = forward(data)
forward.dump("model.mgb", arg_names=["data"])
"""


def format_code(code, indent=0):
    c = code
    if indent:
        c = [" " * indent + x for x in code]
    return "\n".join(c)


def main():
    from torchvision.models.resnet import resnet18

    net = resnet18(pretrained=True)

    with open("state.pkl", "wb") as f:
        state = {k: v.numpy() for k, v in net.state_dict().items()}
        pickle.dump(state, f)

    symbolic_traced: torch.fx.GraphModule = symbolic_trace(net)

    ctx = VisitorContext()
    visit_module("root", symbolic_traced, ctx)
    code = convert_code(symbolic_traced)

    with open("code.py", "w") as f:
        f.write(
            template.format(
                module_code=format_code(ctx.code),
                forward_code=format_code(code, indent=4),
            )
        )


if __name__ == "__main__":
    main()
