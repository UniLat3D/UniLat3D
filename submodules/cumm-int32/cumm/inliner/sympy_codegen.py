import abc
import sympy 
from sympy.printing.c import C99CodePrinter
from sympy.codegen.ast import float32, real
sympy.Integer
from functools import cache, partial


import sys
from typing import Any, Callable, Mapping, Sequence
from pydantic import field_validator, model_validator
from typing_extensions import Self 
import typing 
import types
from typing import Optional, Type 
from typing_extensions import Literal, Annotated, NotRequired, get_origin, get_args, get_type_hints
from dataclasses import dataclass
import inspect 
from sympy.core import S

from ccimport import compat
def lenient_issubclass(cls: Any,
                       class_or_tuple: Any) -> bool:  # pragma: no cover
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)

def is_annotated(ann_type: Any) -> bool:
    # https://github.com/pydantic/pydantic/blob/35144d05c22e2e38fe093c533ff3a05ce9a30116/pydantic/_internal/_typing_extra.py#L99C1-L104C1
    origin = get_origin(ann_type)
    return origin is not None and lenient_issubclass(origin, Annotated)
if sys.version_info < (3, 10):

    def origin_is_union(tp: Optional[Type[Any]]) -> bool:
        return tp is typing.Union

else:
    def origin_is_union(tp: Optional[Type[Any]]) -> bool:
        return tp is typing.Union or tp is types.UnionType  # noqa: E721

def is_optional(ann_type: Any) -> bool:
    origin = get_origin(ann_type)
    return origin is not None and origin_is_union(origin) and type(None) in get_args(ann_type)


@dataclass
class AnnotatedArg:
    name: str 
    param: Optional[inspect.Parameter] 
    type: Any 
    annometa: Optional[tuple[Any, ...]] = None 

@dataclass
class AnnotatedReturn:
    type: Any 
    annometa: Optional[tuple[Any, ...]] = None 

def extract_annotated_type_and_meta(ann_type: Any) -> tuple[Any, Optional[Any]]:
    if is_annotated(ann_type):
        annometa = ann_type.__metadata__
        ann_type = get_args(ann_type)[0]
        return ann_type, annometa
    return ann_type, None

def parse_annotated_function(func: Callable, is_dynamic_class: bool = False) -> tuple[list[AnnotatedArg], Optional[AnnotatedReturn]]:
    if compat.Python3_10AndLater:
        annos = get_type_hints(func, include_extras=True)
    else:
        annos = get_type_hints(func, include_extras=True, globalns={} if is_dynamic_class else None)
    
    specs = inspect.signature(func)
    name_to_parameter = {p.name: p for p in specs.parameters.values()}
    anno_args: list[AnnotatedArg] = []
    return_anno: Optional[AnnotatedReturn] = None
    for name, anno in annos.items():
        if name == "return":
            anno, annotated_metas = extract_annotated_type_and_meta(anno)
            return_anno = AnnotatedReturn(anno, annotated_metas)
        else:
            param = name_to_parameter[name]
            anno, annotated_metas = extract_annotated_type_and_meta(anno)

            arg_anno = AnnotatedArg(name, param, anno, annotated_metas)
            anno_args.append(arg_anno)
    for name, param in name_to_parameter.items():
        if name not in annos and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            anno_args.append(AnnotatedArg(name, param, Any))
    return anno_args, return_anno

class Vector:
    def __init__(self, num: int):
        self.num = num 

class Scalar:
    pass

class VectorExpr:
    def __init__(self, sym: sympy.Expr, num: int):
        self.sym = sym
        self.num = num 

@dataclass
class _SymArg:
    name: str 
    sym: sympy.Symbol
    is_vector: bool
    count: int = 1

@dataclass
class _SymRet:
    name: str 
    sym: sympy.Expr
    is_vector: bool
    count: int = 1


def _pow_is_y_integer(x, y):
    return y.is_integer and int(y) >= 0

def _pow_is_y_negative_integer(x, y):
    return y.is_integer and (int(y) <= -1)

def _pow_common_optim_pow_positive(x, y, use_f64_suffix):
    if int(y) == 0:
        return "1" if use_f64_suffix else "1.0f"
    assert int(y) > 0
    pow_optim = '*'.join(['(' + x + ')']*int(y))
    return f"({pow_optim})"


def _pow_common_optims_negative(x, y, prefix, use_f64_suffix):
    pow_optim = '*'.join(['(' + x + ')']*abs(int(y)))
    if use_f64_suffix:
        return f"1.0 / ({pow_optim})"
    else:
        return f"1.0f / ({pow_optim})"


class TvArrayCodePrinter(C99CodePrinter):
    def __init__(self, prefix="math_op_t", use_f64_suffix: bool = False) -> None:
        settings = {}
        if not use_f64_suffix:
            settings["type_aliases"] = {
                real: float32
            }
        super().__init__(settings)
        self.known_functions = {k: f"{prefix}::{v}" for k, v in self.known_functions.items()}
        # self.known_functions["Pow"] = "POW"
        self.known_functions["Pow"] = [
            (_pow_is_y_integer, partial(_pow_common_optim_pow_positive, use_f64_suffix=use_f64_suffix)),
            (_pow_is_y_negative_integer, partial(_pow_common_optims_negative, prefix=prefix, use_f64_suffix=use_f64_suffix)),
            (lambda x, y: not y.is_integer, lambda x, y: f"{prefix}::pow({x}, {y})")]
        if not use_f64_suffix:
            self.math_macros[S.Pi] = "static_cast<float>(M_PI)"

def tvarray_math_expr(expr, prefix: str = "math_op_t", use_f64_suffix: bool = False):
    res = TvArrayCodePrinter(prefix, use_f64_suffix).doprint(expr)
    assert isinstance(res, str)
    return res 

def sigmoid(x):
    return 1.0 / (1.0 + sympy.exp(-x))

def logit(x):
    return sympy.log(x / (1.0 - x))

def subs_vector(expr, vecs: list[sympy.Symbol], index: int):
    return expr.subs([[vec, sympy.Symbol(f"{vec.name}[{index}]")] for vec in vecs])

class VectorSymOperator(abc.ABC):
    """No reduction, no array substitution/reorder Vector Expr Code Generator
    No reduction, no array substitution/reorder means your expr
    can't have any reduction operation, and array substitution.
    """
    Scalar = Scalar 
    Vector = Vector
    def __init__(self, is_double: bool = False, prefix: str = "math_op_t"):
        self.name_to_sym: dict[str, _SymArg] = {}
        self.name_to_res_sym: dict[str, _SymRet] = {}
        self._is_built = False
        self.is_double = is_double
        self._prefix = prefix

        self._cached_generate_result_expr_code = {}
        self._cached_generate_gradients_code = {}

    def build(self):
        if self._is_built:
            return self
        self.name_to_sym, self.name_to_res_sym = self._run_sympy_forward()
        self._is_built = True
        return self

    @abc.abstractmethod
    def forward(self, *args) -> dict[str, sympy.Expr | VectorExpr]:
        raise NotImplementedError

    def generate_result_expr_code(self, name: str):
        if name in self._cached_generate_result_expr_code:
            return self._cached_generate_result_expr_code[name]
        assert self._is_built, "you must call build() before generating code"
        vec_inputs_list = [v.sym for v in self.name_to_sym.values() if v.is_vector]
        sym_ret = self.name_to_res_sym[name]
        if sym_ret.is_vector:
            res_exprs: list[str] = []
            for i in range(sym_ret.count):
                expr_str = subs_vector(sym_ret.sym, vec_inputs_list, i)
                res_exprs.append(tvarray_math_expr(expr_str, self._prefix))
            if self.is_double:
                dtype_str = "double"
            else:
                dtype_str = "float"
            res = f"tv::array<{dtype_str}, {sym_ret.count}>{{{', '.join(res_exprs)}}}"
        else:
            # assume no reduction, and no array substitution
            # so we can directly return the expression
            res = tvarray_math_expr(sym_ret.sym, self._prefix)
        self._cached_generate_result_expr_code[name] = res
        return res

    def generate_gradients_code(self, inp_name: str, out_grad_name_dict: Mapping[str, str | None], simplify: bool = True):
        if inp_name in self._cached_generate_gradients_code:
            return self._cached_generate_gradients_code[inp_name]
        assert self._is_built, "you must call build() before generating code"
        vec_inputs_list = [v.sym for v in self.name_to_sym.values() if v.is_vector]
        for v in out_grad_name_dict.values():
            assert v not in self.name_to_sym, f"key {v} already exists in args, use another key"
            assert v not in self.name_to_res_sym, f"key {v} already exists in return, use another key"
        sym_arg_obj = self.name_to_sym[inp_name]
        if not sym_arg_obj.is_vector:
            res_expr_syms: list[sympy.Expr] = []
            for out_v in self.name_to_res_sym.values():
                out_v_grad_name = out_grad_name_dict[out_v.name]
                if out_v_grad_name is not None:
                    if out_v.is_vector:
                        for i in range(out_v.count):
                            grad_expr = sympy.diff(out_v.sym, sym_arg_obj.sym)
                            grad_expr = subs_vector(grad_expr, vec_inputs_list, i)
                            res_expr_syms.append(grad_expr * sympy.Symbol(f"{out_v_grad_name}[{i}]"))
                    else:
                        grad_expr = sympy.diff(out_v.sym, sym_arg_obj.sym)
                        res_expr_syms.append(grad_expr * sympy.Symbol(out_v_grad_name))
            res_expr = sum(res_expr_syms)
            res_expr = sympy.simplify(res_expr) if simplify else res_expr
            res_expr_code = tvarray_math_expr(res_expr, self._prefix)
            res = res_expr_code
        else:
            res_exprs: list[str] = []

            for j in range(sym_arg_obj.count):
                res_expr_syms: list[sympy.Expr] = []
                for out_v in self.name_to_res_sym.values():
                    out_v_grad_name = out_grad_name_dict[out_v.name]
                    if out_v_grad_name is not None:
                        if out_v.is_vector:
                            grad_expr = sympy.diff(out_v.sym, sym_arg_obj.sym)
                            grad_expr = subs_vector(grad_expr, vec_inputs_list, j)
                            res_expr_syms.append(grad_expr * sympy.Symbol(f"{out_v_grad_name}[{j}]"))
                        else:
                            grad_expr = sympy.diff(out_v.sym, sym_arg_obj.sym)
                            res_expr_syms.append(grad_expr * sympy.Symbol(out_v_grad_name))
                res_expr = sum(res_expr_syms)
                res_expr = sympy.simplify(res_expr) if simplify else res_expr
                res_exprs.append(tvarray_math_expr(res_expr, self._prefix))
            if self.is_double:
                dtype_str = "double"
            else:
                dtype_str = "float"
            res = f"tv::array<{dtype_str}, {sym_arg_obj.count}>{{{', '.join(res_exprs)}}}"
        self._cached_generate_gradients_code[inp_name] = res
        return res

    def generate_gradients_symbols(self, inp_name: str, out_grad_name_dict: dict[str, str | None], simplify: bool = True):
        assert self._is_built, "you must call build() before generating code"
        vec_inputs_list = [v.sym for v in self.name_to_sym.values() if v.is_vector]
        for v in out_grad_name_dict.values():
            assert v not in self.name_to_sym, f"key {v} already exists in args, use another key"
            assert v not in self.name_to_res_sym, f"key {v} already exists in return, use another key"
        sym_arg_obj = self.name_to_sym[inp_name]
        res: list[sympy.Expr] = []
        if not sym_arg_obj.is_vector:
            res_expr_syms: list[sympy.Expr] = []
            for out_v in self.name_to_res_sym.values():
                out_v_grad_name = out_grad_name_dict[out_v.name]
                if out_v_grad_name is not None:
                    if out_v.is_vector:
                        for i in range(out_v.count):
                            grad_expr = sympy.diff(out_v.sym, sym_arg_obj.sym)
                            grad_expr = subs_vector(grad_expr, vec_inputs_list, i)
                            res_expr_syms.append(grad_expr * sympy.Symbol(f"{out_v_grad_name}[{i}]"))
                    else:
                        grad_expr = sympy.diff(out_v.sym, sym_arg_obj.sym)
                        res_expr_syms.append(grad_expr * sympy.Symbol(out_v_grad_name))
            res_expr = sum(res_expr_syms)
            res.append(sympy.simplify(res_expr) if simplify else res_expr)
        else:

            for j in range(sym_arg_obj.count):
                res_expr_syms: list[sympy.Expr] = []
                for out_v in self.name_to_res_sym.values():
                    out_v_grad_name = out_grad_name_dict[out_v.name]
                    if out_v_grad_name is not None:
                        if out_v.is_vector:
                            grad_expr = sympy.diff(out_v.sym, sym_arg_obj.sym)
                            grad_expr = subs_vector(grad_expr, vec_inputs_list, j)
                            res_expr_syms.append(grad_expr * sympy.Symbol(f"{out_v_grad_name}[{j}]"))
                        else:
                            grad_expr = sympy.diff(out_v.sym, sym_arg_obj.sym)
                            res_expr_syms.append(grad_expr * sympy.Symbol(out_v_grad_name))
                res_expr = sum(res_expr_syms)
                res.append(sympy.simplify(res_expr) if simplify else res_expr)
        return res


    def _parse_forward(self):
        args, _ = parse_annotated_function(self.forward)
        sym_args: list[AnnotatedArg] = []
        for arg in args:
            metadata = arg.annometa
            # if metadata is None, treat as scalar
            if metadata is None or len(metadata) == 0:
                metadata = (Scalar(),)
                arg.annometa = metadata
            assert metadata is not None and len(metadata) == 1
            meta = metadata[0]
            assert isinstance(meta, (Vector, Scalar)), "you must annotate all args with Vector or Scalar"
            sym_args.append(arg)
        return sym_args

    def _run_sympy_forward(self):
        args = self._parse_forward()
        name_to_sym: dict[str, _SymArg] = {}
        args_sym = []
        for arg in args:
            sym_arg = sympy.Symbol(arg.name)
            assert arg.annometa is not None 
            if isinstance(arg.annometa[0], Vector):
                sym_arg_obj = _SymArg(arg.name, sym_arg, True, arg.annometa[0].num)
            else:
                sym_arg_obj = _SymArg(arg.name, sym_arg, False)
            name_to_sym[arg.name] = sym_arg_obj
            args_sym.append(sym_arg)
        forward_result = self.forward(*args_sym)
        name_to_res_sym: dict[str, _SymRet] = {}
        for k, v in forward_result.items():
            assert k not in name_to_sym, f"key {k} already exists in args, use another key"
            if isinstance(v, VectorExpr):
                sym_res_obj = _SymRet(k, v.sym, True, v.num)
            else:
                sym_res_obj = _SymRet(k, v, False)
            name_to_res_sym[k] = sym_res_obj
        return name_to_sym, name_to_res_sym



class VectorSymExpr:
    def __init__(self, expr, vecs: list[sympy.Symbol]):
        self.expr = expr
        self.vecs = vecs

    def __getitem__(self, index):
        return subs_vector(self.expr, self.vecs, index) 

def __main():
    class Test(VectorSymOperator):
        def forward(self, x: Annotated[sympy.Symbol, Vector(3)], y: Annotated[sympy.Symbol, Scalar()]) -> dict[str, VectorExpr]:
            return {
                "z": VectorExpr(x + y, 3)
            }
    class TestComplex(VectorSymOperator):
        def forward(self, t_gaussian: Annotated[sympy.Symbol, Scalar()], velocity: Annotated[sympy.Symbol, Vector(3)], scale_t: Annotated[sympy.Symbol, Scalar()], batch_t: Annotated[sympy.Symbol, Scalar()], _proxy_xyz: Annotated[sympy.Symbol, Vector(3)], _proxy_opacity_val: Annotated[sympy.Symbol, Scalar()], time_shift: Annotated[sympy.Symbol, Scalar()]) -> dict[str, VectorExpr]:
            a = 1.0 / 0.2 * sympy.pi * 2.0
            scale_t_act = sympy.exp(scale_t)
            xyz_shm = _proxy_xyz + velocity * sympy.sin((batch_t - t_gaussian) * a) / a
            inst_velocity = velocity * sympy.exp(-scale_t_act / 0.2 / 2.0 * 1.0)
            xyz_shm += inst_velocity * time_shift
            marginal_t = sympy.exp(-0.5 * (t_gaussian - batch_t) * (t_gaussian - batch_t) / scale_t_act / scale_t_act)
            opacity_t = sigmoid(_proxy_opacity_val) * marginal_t
            return {
                "xyz": VectorExpr(xyz_shm, 3),
                "opacity": opacity_t,
            }

    x, y, z = sympy.symbols('x[0] y z')
    x2, y2, z2 = sympy.symbols('x2 0.5 z2')

    expr = sympy.Add(x, y)
    print(expr.func, sympy.Add)
    expr1 = sympy.exp(1.0 / (x*x) + sigmoid(x) + sympy.Float(0.5))
    expr = sympy.diff(expr1, x)
    # print(sympy.diff(sympy.exp(x**2 + 0.5), x))
    for arg in sympy.preorder_traversal(expr):
        print(arg, type(arg))
    print(expr1)
    print(tvarray_math_expr(expr1))
    print(expr)
    print(tvarray_math_expr(expr))
    # print(sympy.Matrix([[x, y, z], [z2, y2, x2]])[1, 0])

    # print(sympy.exp(vector("x0", "x1")))

    test = Test().build()
    print(test.generate_result_expr_code("z"))
    print(test.generate_gradients_code("y", {"z": "grad_z"}))

    test_complex = TestComplex().build()
    print(test_complex.generate_result_expr_code("xyz"))
    print(test_complex.generate_gradients_code("t_gaussian", {
        "xyz": "grad_xyz",
        "opacity": "grad_opacity"
    }))



if __name__ == "__main__":
    __main()