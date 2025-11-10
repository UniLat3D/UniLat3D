from cumm import tensorview as tv 
from typing import List, Optional, Tuple, Union
import numpy as np 

class TensorCreator:
    def __init__(self, dtype: int, device: int) -> None:
        self.dtype = dtype
        self.device = device

    def zeros(
            self,
            *shapes: Union[List[int], Tuple[int, ...]],
            dtype: Optional[int] = None,
            device: Optional[int] = None,
            shared_buffer: bool = False) -> Tuple[tv.Tensor, ...]:
        if dtype is None:
            dtype = self.dtype

        if device is None:
            device = self.device
        if shared_buffer:
            total_size = 0
            sizes: List[int] = []
            for s in shapes:
                sizes.append(np.prod(s))
                total_size += sizes[-1]
            buffer = tv.zeros([total_size], dtype=dtype, device=device)
            res: List[tv.Tensor] = []
            start = 0
            for s, size in zip(shapes, sizes):
                res.append(buffer[start:start+size].view(list(s)))
                start += size 
            return tuple(res)
        return tuple(
            tv.zeros(list(s), dtype=dtype, device=device) for s in shapes)

    def empty(
            self,
            *shapes: Union[List[int], Tuple[int, ...]],
            dtype: Optional[int] = None,
            device: Optional[int] = None,
            shared_buffer: bool = False) -> Tuple[tv.Tensor, ...]:
        if dtype is None:
            dtype = self.dtype

        if device is None:
            device = self.device
        if shared_buffer:
            total_size = 0
            sizes: List[int] = []
            for s in shapes:
                sizes.append(np.prod(s))
                total_size += sizes[-1]
            buffer = tv.empty([total_size], dtype=dtype, device=device)
            res: List[tv.Tensor] = []
            start = 0
            for s, size in zip(shapes, sizes):
                res.append(buffer[start:start+size].view(list(s)))
                start += size 
            return tuple(res)
        return tuple(
            tv.empty(list(s), dtype=dtype, device=device) for s in shapes)

    def to(self, dtype: int):
        return TensorCreator(dtype, self.device)
    
    def float(self):
        return self.to(tv.float32)
    
    def double(self):
        return self.to(tv.float64)

    def int(self):
        return self.to(tv.int32)

    def long(self):
        return self.to(tv.int64)

    def char(self):
        return self.to(tv.int8)

    def uint8(self):
        return self.to(tv.uint8)
