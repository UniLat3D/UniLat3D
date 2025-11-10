from typing import Any, Union
from typing import Dict, Hashable
import numpy as np
from cumm import tensorview as tv
import json
from collections import abc
from functools import reduce 

JSON_INDEX_KEY = "__cumm_io_json_index"

NPDTYPE_TO_JSONARRAY_MAP = {
    np.dtype(np.uint64): tv.uint64,
    np.dtype(np.uint32): tv.uint32,
    np.dtype(np.uint16): tv.uint16,
    np.dtype(np.uint8): tv.uint8,
    np.dtype(np.int64): tv.int64,
    np.dtype(np.int32): tv.int32,
    np.dtype(np.int16): tv.int16,
    np.dtype(np.int8): tv.int8,
    np.dtype(np.float64): tv.float64,
    np.dtype(np.float32): tv.float32,
    np.dtype(np.float16): tv.float16,
    np.dtype(np.bool_): tv.bool_,
}


def _inv_map(dict_map: Dict[Hashable, Hashable]) -> Dict[Hashable, Hashable]:
    return {v: k for k, v in dict_map.items()}


INV_NPDTYPE_TO_JSONARRAY_MAP = _inv_map(NPDTYPE_TO_JSONARRAY_MAP)


class Placeholder(object):
    def __init__(self, index: int, nbytes: int):
        self.index = index
        self.nbytes = nbytes

    def __add__(self, other):
        assert self.index == other.index
        return Placeholder(self.index, self.nbytes + other.nbytes)

    def __repr__(self):
        return "Placeholder[{},{}]".format(self.index, self.nbytes)

    def __eq__(self, other):
        return self.index == other.index and self.nbytes == other.nbytes


def is_json_index(data):
    return isinstance(data, dict) and JSON_INDEX_KEY in data


def byte_size(obj: Union[np.ndarray, tv.Tensor]) -> int:
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, tv.Tensor):
        return obj.size * obj.itemsize
    else:
        raise NotImplementedError


def _extract_arrays_from_data(arrays,
                              data,
                              object_classes=(np.ndarray,),
                              json_index=False):
    # can't use abc.Sequence because string is sequence too.
    if isinstance(data, (list, tuple)):
        data_skeleton = [None] * len(data)
        for i in range(len(data)):
            e = data[i]
            if isinstance(e, object_classes):
                data_skeleton[i] = {JSON_INDEX_KEY: len(arrays)}
                arrays.append(e)
            else:
                data_skeleton[i] = _extract_arrays_from_data(
                    arrays, e, object_classes, json_index)
        if isinstance(data, tuple):
            data_skeleton = tuple(data_skeleton)
        return data_skeleton
    elif isinstance(data, abc.Mapping):
        data_skeleton = {}
        for k, v in data.items():
            if isinstance(v, object_classes):
                data_skeleton[k] = {JSON_INDEX_KEY: len(arrays)}
                arrays.append(v)
            else:
                data_skeleton[k] = _extract_arrays_from_data(
                    arrays, v, object_classes, json_index)
        return data_skeleton
    else:
        data_skeleton = None
        if isinstance(data, object_classes):
            data_skeleton = {JSON_INDEX_KEY: len(arrays)}
            arrays.append(data)
        else:
            data_skeleton = data
        return data_skeleton


def extract_arrays_from_data(data,
                             object_classes=(np.ndarray,),
                             json_index=False):
    arrays = []
    data_skeleton = _extract_arrays_from_data(arrays,
                                              data,
                                              object_classes=object_classes,
                                              json_index=json_index)
    return arrays, data_skeleton


def align_offset(offset, n):
    """given a byte offset, align it and return an aligned offset
    """
    if n <= 0:
        return offset
    return n * ((offset + n - 1) // n)


def put_arrays_to_data(arrays, data_skeleton, json_index=False) -> Any:
    if not arrays:
        return data_skeleton
    return _put_arrays_to_data(arrays, data_skeleton, json_index)


def _put_arrays_to_data(arrays, data_skeleton, json_index=False):
    if isinstance(data_skeleton, (list, tuple)):
        length = len(data_skeleton)
        data = [None] * length
        for i in range(length):
            e = data_skeleton[i]
            if is_json_index(e):
                data[i] = arrays[e[JSON_INDEX_KEY]]
            else:
                data[i] = _put_arrays_to_data(arrays, e, json_index)
        if isinstance(data_skeleton, tuple):
            data = tuple(data)
        return data
    elif isinstance(data_skeleton, abc.Mapping):
        data = {}
        for k, v in data_skeleton.items():
            if is_json_index(v):
                data[k] = arrays[v[JSON_INDEX_KEY]]
            else:
                data[k] = _put_arrays_to_data(arrays, v, json_index)
        return data
    else:
        if is_json_index(data_skeleton):
            data = arrays[data_skeleton[JSON_INDEX_KEY]]
        else:
            data = data_skeleton
        return data


def dumps_jsonarray(obj, multi_thread=False, buffer=None, use_bytearray=False, align_size: int = 32):
    """
    layout:
    +--------------+------------+---------------------------------+--------------+
    |meta_start_pos|meta_end_pos|      array/bytes content        |     meta     |
    +--------------+------------+---------------------------------+--------------+
    data without array/bytes will be saved as bytes in content.
    meta format:
    {
        "array": [
            {
                "shape": [...]
                "dtype": ...
                "offset": ...
            }
        ]
        "data": skeleton
    }
    """
    arrays, data_skeleton = extract_arrays_from_data(obj, (np.ndarray, tv.Tensor), True)
    array_meta = []
    start = 16
    for i in range(len(arrays)):
        arr = arrays[i]
        start_aligned = align_offset(start, align_size)
        if isinstance(arr, tv.Tensor):
            assert arr.device == -1
            arr_np = arr.numpy_view()
        else:
            arr_np = arr
        # ascontiguous will convert scalar to 1-D array. be careful.
        arrays[i] = np.ascontiguousarray(arr_np)
        array_meta.append({
            "shape": arrays[i].shape,
            "dtype": NPDTYPE_TO_JSONARRAY_MAP[arrays[i].dtype],
            "offset": start_aligned,
            "is_np": isinstance(arr, np.ndarray),
        })
        start = start_aligned + arrays[i].nbytes
    meta = {
        "array": array_meta,
        "data": data_skeleton,
    }
    meta_json = json.dumps(meta).encode("utf8")
    meta_length = len(meta_json)
    array_buffers = []
    for i in range(len(arrays)):
        array_buffers.append((arrays[i].view(np.uint8),
                              array_meta[i]["offset"], arrays[i].nbytes))

    total_length = start + meta_length
    if buffer is None:
        if not use_bytearray:
            buffer = np.empty(total_length, dtype=np.uint8)
        else:
            buffer = bytearray(total_length)
    else:
        assert len(buffer) >= total_length
    buffer_view = memoryview(buffer)
    content_end_offset = start
    meta_end_offset = content_end_offset + meta_length
    buffer_view[:8] = np.array(content_end_offset, dtype=np.int64).tobytes()
    buffer_view[8:16] = np.array(meta_end_offset, dtype=np.int64).tobytes()
    buffer_view[16:24] = np.array(align_size, dtype=np.int64).tobytes()

    shared_mem = np.frombuffer(buffer_view, dtype=np.uint8)
    for a_buf, offset, size in array_buffers:
        shared_mem_view = memoryview(shared_mem[offset:offset + size])
        if not isinstance(a_buf, bytes):
            buf_mem_view = memoryview(a_buf.reshape(-1))
            if multi_thread:  # slow when multi_thread copy in worker
                shared_mem[offset:offset + size] = a_buf.reshape(-1)
            else:
                shared_mem_view[:] = buf_mem_view
        else:
            shared_mem_view[:] = a_buf

    shared_mem[content_end_offset:content_end_offset +
               meta_length] = np.frombuffer(meta_json, dtype=np.uint8)
    return buffer


def loads_jsonarray(binary, copy=True):
    buffer_view = memoryview(binary)
    content_end_offset = np.frombuffer(buffer_view[:8], dtype=np.int64).item()
    meta_end_offset = np.frombuffer(buffer_view[8:16], dtype=np.int64).item()
    pb_bytes = buffer_view[content_end_offset:meta_end_offset]
    meta = json.loads(bytearray(pb_bytes))
    array_metas = meta["array"]
    data_skeleton = meta["data"]
    shared_mem = buffer_view
    results_array = []
    for array_meta in array_metas:
        shape = array_meta["shape"]
        dtype = INV_NPDTYPE_TO_JSONARRAY_MAP[array_meta["dtype"]]
        offset = array_meta["offset"]
        is_np = array_meta["is_np"]

        length = reduce(lambda x, y: x * y, shape) * np.dtype(dtype).itemsize
        arr = np.frombuffer(memoryview(shared_mem[offset:offset + length]),
                            dtype=dtype).reshape(shape)
        if is_np:
            if copy:
                arr = arr.copy()
        else:
            arr = tv.from_numpy(arr)
            if copy:
                arr = arr.clone()
        results_array.append(arr)
    results = put_arrays_to_data(results_array, data_skeleton, json_index=True)
    return results

