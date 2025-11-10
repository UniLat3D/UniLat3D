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

import functools
from typing import Generic, List, Sequence, TypeVar, Union, overload

T = TypeVar('T')


class MetaArray(Generic[T]):
    def __init__(self, *values: T, check_floor_div: bool = True):
        self.data: List[T] = [*values]
        self.size = len(values)
        self._check_floor_div = check_floor_div

    def __len__(self):
        return self.size

    def __setitem__(self, idx: int, val: T):
        self.data[idx] = val

    def __add__(self, other: "MetaArray[T]") -> "MetaArray[T]":
        assert self.size == other.size, f"size mismatch, {self.size} vs {other.size}"
        return MetaArray(*[x + y for x, y in zip(self.data, other.data)],
                         check_floor_div=self._check_floor_div)

    def __sub__(self, other: "MetaArray[T]") -> "MetaArray[T]":
        assert self.size == other.size, f"size mismatch, {self.size} vs {other.size}"
        return MetaArray(*[x - y for x, y in zip(self.data, other.data)],
                         check_floor_div=self._check_floor_div)

    def __mul__(self, other: "MetaArray[T]") -> "MetaArray[T]":
        assert self.size == other.size, f"size mismatch, {self.size} vs {other.size}"
        return MetaArray(*[x * y for x, y in zip(self.data, other.data)],
                         check_floor_div=self._check_floor_div)

    def __truediv__(self, other: "MetaArray[T]") -> "MetaArray[float]":
        assert self.size == other.size, f"size mismatch, {self.size} vs {other.size}"
        return MetaArray(*[x / y for x, y in zip(self.data, other.data)],
                         check_floor_div=self._check_floor_div)

    def __floordiv__(self, other: "MetaArray[T]") -> "MetaArray[T]":
        assert self.size == other.size, f"size mismatch, {self.size} vs {other.size}"
        if self._check_floor_div:
            for i in range(self.size):
                assert self[i] % other[i] == 0, "meta array must divible"
        return MetaArray(*[x // y for x, y in zip(self.data, other.data)],
                         check_floor_div=self._check_floor_div)

    @overload
    def __getitem__(self, obj: int) -> T:
        ...

    @overload
    def __getitem__(self, obj: slice) -> "MetaArray[T]":
        ...

    def __getitem__(self, obj):
        if isinstance(obj, slice):
            return MetaArray(*self.data[obj],
                             check_floor_div=self._check_floor_div)
        return self.data[obj]

    def __eq__(self, other: Union["MetaArray[T]", Sequence[T]]):
        assert self.size == len(
            other), f"size mismatch, {self.size} vs {len(other)}"
        if isinstance(other, MetaArray):
            return all(x == y for x, y in zip(self.data, other.data))
        else:
            return all(x == y for x, y in zip(self.data, other))

    def __ne__(self, other: Union["MetaArray[T]", Sequence[T]]):
        return not self == other

    def prod(self) -> T:
        return functools.reduce(lambda x, y: x * y, self.data)

    def __repr__(self) -> str:
        return str(self.data)

    def copy(self) -> "MetaArray[T]":
        return MetaArray(*self.data, check_floor_div=self._check_floor_div)

    def __iter__(self):
        yield from self.data


def seq(*value: T) -> MetaArray[T]:
    return MetaArray(*value, check_floor_div=False)


def metaseq(*value: T) -> MetaArray[T]:
    """floordiv of metaseq must be divisible
    """
    return MetaArray(*value, check_floor_div=True)


if __name__ == "__main__":
    ma = seq(1, 2, 3)
    for i in ma:
        print(i)
