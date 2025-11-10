# Copyright 2024 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from typing import (Any, Callable, Dict, Generator, Generic, Hashable,
                    Iterable, List, Tuple, TypeVar, Union)

import pccm

from cumm import dtypes

_T = TypeVar("_T")
_T2 = TypeVar("_T2")


def group_by(key_func: Callable[[_T], _T2],
             data_iter: Iterable[_T]) -> Dict[_T2, List[_T]]:
    res = OrderedDict()  # type: OrderedDict[_T2, List[_T]]
    for d in data_iter:
        key = key_func(d)  # type: _T2
        if key not in res:
            res[key] = []
        res[key].append(d)
    return res


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b

def align_up(a: int, b: int) -> int:
    return div_up(a, b) * b



def unpack(arr_name: str,
           indices: Iterable[int],
           op: str = ", ",
           left: str = "[",
           right: str = "]"):
    return op.join(f"{arr_name}{left}{i}{right}" for i in indices)


def unpack_str(arr_name: str,
               indices: Iterable[int],
               op: str = ", ",
               left: str = "_"):
    return unpack(arr_name, indices, op, left, "")


class Condition:
    def __init__(self, flag: bool):
        self.flag = flag

    def __call__(self, true_cond: str, false_cond: str = ""):
        if self.flag:
            return true_cond
        else:
            return false_cond


def dispatch_ints(code: pccm.FunctionCode, ints: List[int], var: str):
    """equivalent to tv::dispatch_int, exists because some compiler
    don't support nested dispatch_int.
    """
    for i, val in enumerate(ints):
        if i == 0:
            with code.if_(f"{var} == {val}"):
                yield val
        else:
            with code.else_if_(f"{var} == {val}"):
                yield val
    with code.else_():
        code.raw(f"""
        TV_THROW_RT_ERR("unknown val {var}, available: {ints}")
        """)


def dispatch(code: pccm.FunctionCode, dts: List[dtypes.DType], var: str):
    """equivalent to tv::dispatch, exists because some compiler
    don't support nested dispatch.
    """
    for i, dtype in enumerate(dts):
        if i == 0:
            with code.if_(f"{var} == tv::DType({dtype.tv_dtype})"):
                yield dtype
        else:
            with code.else_if_(f"{var} == tv::DType({dtype.tv_dtype})"):
                yield dtype
    with code.else_():
        code.raw(f"""
        TV_THROW_RT_ERR("unknown dtype {var}, ", tv::dtype_str({var}), " available: {dts}")
        """)
