from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue, enum
from cumm.tensorview import Tensor
class ArrayPtr:
    dtype_: Any
    length_: int
    access_byte_size_: int
    byte_offset_: int
    align_: int
    itemsize_: int
    data_: Tensor
    meta_data_: Tensor
    def __init__(self, dtype: int, length: int, access_byte_size: int = -1, byte_offset: int = 0, align: int = -1, external_data: Tensor =  Tensor(), meta_data: Tensor =  Tensor()) -> None: 
        """
        Args:
            dtype: 
            length: 
            access_byte_size: 
            byte_offset: 
            align: 
            external_data: 
            meta_data: 
        """
        ...
    def __repr__(self) -> str: ...
    def clear(self) -> None: ...
    def shadow_copy(self) -> "ArrayPtr": ...
    def copy(self) -> "ArrayPtr": ...
    @property
    def length(self) -> int: ...
    @property
    def byte_length(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def access_offset(self) -> int: ...
    @property
    def access_size(self) -> int: ...
    @property
    def num_access(self) -> int: ...
    def __add__(self, index: int) -> "ArrayPtr": 
        """
        Args:
            index: 
        """
        ...
    def __iadd__(self, index: int) -> "ArrayPtr": 
        """
        Args:
            index: 
        """
        ...
    def __sub__(self, index: int) -> "ArrayPtr": 
        """
        Args:
            index: 
        """
        ...
    def __isub__(self, index: int) -> "ArrayPtr": 
        """
        Args:
            index: 
        """
        ...
    def change_access_size(self, new_acc_size: int, align: int = -1) -> "ArrayPtr": 
        """
        Args:
            new_acc_size: 
            align: 
        """
        ...
    def change_access_byte_size(self, new_acc_byte_size: int, align: int = -1) -> "ArrayPtr": 
        """
        Args:
            new_acc_byte_size: 
            align: 
        """
        ...
    def __getitem__(self, idx_access: int) -> "ArrayPtr": 
        """
        Args:
            idx_access: 
        """
        ...
    def __setitem__(self, idx_access: int, val: "ArrayPtr") -> None: 
        """
        Args:
            idx_access: 
            val: 
        """
        ...
    @property
    def data(self) -> Tensor: ...
    @property
    def meta_data(self) -> Tensor: ...
    def astype(self, dtype: int) -> "ArrayPtr": 
        """
        Args:
            dtype: 
        """
        ...
    @staticmethod
    def check_smem_bank_conflicit(idx_ptrs: List[Tuple[int, ArrayPtr]]) -> Dict[int, List[int]]: 
        """
        Args:
            idx_ptrs: 
        """
        ...
