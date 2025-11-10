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

import asyncio
import dataclasses
import enum
from functools import partial
import traceback
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
import tensorpc

from tensorpc.flow import flowapp

from tensorpc.flow import mui, three
import numpy as np
import inspect
from cumm import dtypes
from cumm.gemm import kernel
from cumm.gemm.algospec.core import TensorOp
from cumm.gemm.arch.tensorop import ALL_TENSOR_OP_MAP
from cumm.gemm.core.metaarray import MetaArray, metaseq, seq
from cumm.gemm.constants import SMEM_BANK_SIZE_BITS
import pyee
from . import utils
_TERMINAL_FONT_FAMILY = "IBMPlexMono,SFMono-Regular,Consolas,Liberation Mono,Menlo,Courier,monospace"
from cumm import tensorview as tv
from cumm.gemm.main import GemmAlgoParams, gen_gemm_kernels

def _create_matrix_centers(shape: List[int]):
    centers_scalar = np.arange(shape[0] * shape[1]).astype(np.float32)
    centers = np.stack(
        [-(centers_scalar // shape[1]), centers_scalar % shape[1]], axis=1)
    res = centers[:, ::-1]
    res[:, 0] += 0.5
    res[:, 1] -= 0.5
    return res


class DesignerType(enum.Enum):
    TopViewer = "TopViewer"
    InputIterA = "InputIterA"
    InputIterB = "InputIterB"


@dataclasses.dataclass
class Code:
    code: str 
    code_key: str 

    @classmethod
    def from_dict(cls, state: Dict[str, Any]):
        return cls(state["code"], state["code_key"])

    def to_dict(self):
        return dataclasses.asdict(self)


class WarpPanel(three.Group):
    LaneOver = "laneover"
    LaneOut = "laneout"
    EvWarpOver = "warpover"
    EvWarpOut = "warpout"
    EvQPOver = "qpover"
    EvQPOut = "qpout"
    EvPOver = "pover"
    EvPOut = "pout"

    LaneSelect = "laneselect"
    WarpSelect = "warpselect"


    Reset = "reset"

    def __init__(
        self,
        num_warps,
        size: float,
    ) -> None:
        super().__init__({})
        self.size = size
        self.add_layout(self.create_layout(num_warps, size))
        self.ee = pyee.AsyncIOEventEmitter()

    def create_layout(self, num_warps, size: float):
        layout: Dict[str, Union[three.ToggleButton, three.Button]] = {}
        self.warps_btns: List[three.ToggleButton] = []
        self.warp_btns: List[three.ToggleButton] = []
        warp_colors = [
            "red",
            "blue",
            "green",
            "burlywood",
            "cyan",
            "maroon",
            "chartreuse",
            "blueviolet",
            "gray",
            "orangered",
            "steelblue",
            "palevioletred",
            "palegreen",
            "dodgerblue",
            "deeppink",
            "olive",
            "chocolate",
            "springgreen",
            "seagreen",
        ]
        # TODO add P/QP buttons
        for i in range(num_warps):
            x_idx = i % 8
            y_idx = i // 8

            x = size * 1.05 * x_idx
            y = -size * 1.05 * y_idx
            color = warp_colors[i % len(warp_colors)]
            btn = three.ToggleButton(f"W{i}", size, size,
                                     partial(self.on_warp_select, index=i))
            btn.mesh.set_pointer_callback(
                on_over=three.EventHandler(partial(self.on_warp_over,
                                                    index=i)),
                on_out=three.EventHandler(partial(self.on_warp_out, index=i)))
            btn.props.position = (x, y, 0)
            layout[f"warp_{i}"] = btn
            self.warps_btns.append(btn)

        offset = size * 1.05 * (tv.div_up(num_warps, 8))
        self._lane_colors = np.zeros([32, 3], np.float32)

        for i in range(4):
            for j in range(8):
                idx = i * 8 + j
                color = warp_colors[idx % len(warp_colors)]
                self._lane_colors[idx] = mui.colors.str_to_rgb_float(color)
                btn = three.ToggleButton(
                    str(idx), size, size,
                    partial(self.on_lane_select, index=idx))

                btn.mesh.set_pointer_callback(on_over=three.EventHandler(
                    partial(self.on_lane_over, index=idx)),
                                              on_out=three.EventHandler(
                                                  partial(self.on_lane_out,
                                                          index=idx)))

                btn.props.position = (j * size * 1.05,
                                      -i * size * 1.05 - offset, 0)
                layout[f"lane_{i}_{j}"] = btn
                btn.mesh.props.click_color = color
                self.warp_btns.append(btn)
        btn = three.Button("R", size, size, self.reset)
        btn.props.position = (0 * size * 1.05, -4 * size * 1.05 - offset, 0)
        layout[f"reset"] = btn
        return layout

    def on_warp_select(self, enable, index):
        return self.ee.emit(self.WarpSelect, enable, index)

    def on_lane_select(self, enable, index):
        return self.ee.emit(self.LaneSelect, enable, index)

    def on_lane_over(self, ev, index):
        return self.ee.emit(self.LaneOver, index)

    def on_lane_out(self, ev, index):
        return self.ee.emit(self.LaneOut, index)

    def on_warp_over(self, ev, index):
        return self.ee.emit(self.LaneOver, index)

    def on_warp_out(self, ev, index):
        return self.ee.emit(self.LaneOut, index)

    def is_warp_toggled(self, i: int):
        return self.warps_btns[i].toggled

    def is_lane_toggled(self, i: int):
        return self.warp_btns[i].toggled

    async def reset(self, ev):
        appev = flowapp.AppEvent("", {})
        for v in self.warp_btns:
            appev += v.mesh.update_event(toggled=False)
            # await v.mesh.set_checked(False)
        for v in self.warps_btns:
            appev += v.mesh.update_event(toggled=False)
            # await v.mesh.set_checked(False)
        await self.send_and_wait(appev)
        self.ee.emit(self.Reset)

    async def reset_num_warps(self, num_warps: int):
        await self.set_new_layout(self.create_layout(num_warps, self.size))

    def get_lane_colors_rgba(self, opacity: float):
        opac = np.full([32, 1], opacity, np.float32)
        return np.concatenate([self._lane_colors, opac], axis=1)


def _get_dtype_select(label: str, default: dtypes.DType):
    items = []
    for dt in dtypes.ALL_DTYPES_FOR_GEMM:
        items.append((dt.shortcut(), dt.tv_dtype))
    res = mui.Select(label, items)
    res.props.value = default.tv_dtype
    res.props.mui_margin = "dense"

    return res



class GemmParams(mui.FlexBox):
    OnAnyChanged = "on_any_change"
    OnTileShapeChanged = "on_ts_change"
    OnWarpTileShapeChanged = "on_wts_change"
    OnWarpNumChanged = "on_warp_num_change"

    OnInputParamAChanged = "on_inp_a_param_change"
    OnInputParamBChanged = "on_inp_b_param_change"

    OnTensorOpChanged = "on_top_change"

    TileShape = "ts"
    WarpTileShape = "wts"
    DtypeAB = "dtab"
    DtypeAcc = "dtacc"
    DtypeComp = "dtcomp"
    DtypeC = "dtc"
    TransA = "ta"
    TransB = "tb"
    LdsmA = "ldsma"
    LdsmB = "ldsmb"

    AccessA = "acca"
    AccessB = "accb"
    TensorOp = "top"

    def __init__(self, errors: "Errors") -> None:
        super().__init__()

        self._dtypes_ab = _get_dtype_select("Dtype AB", dtypes.float16)
        self._dtypes_acc = _get_dtype_select("Dtype Acc", dtypes.float16)
        self._dtypes_comp = _get_dtype_select("Dtype Comp", dtypes.float16)
        self._dtypes_c = _get_dtype_select("Dtype C", dtypes.float16)

        self._trans_a = mui.Switch("TransA")
        self._trans_b = mui.Switch("TransB")
        self._ldsm_a = mui.Switch("LdmatrixA").prop(checked=True)
        self._ldsm_b = mui.Switch("LdmatrixB").prop(checked=True)

        self.errors = errors
        # self._trans_c = mui.Switch("TransC")

        self._tile_shape = mui.Input(
            "Tile Shape", init="[128, 128, 32]").prop(mui_margin="dense")
        self._warp_tile_shape = mui.Input(
            "Warp Tile Shape", init="[32, 64, 32]").prop(mui_margin="dense")
        self._tile_shape.prop(padding_bottom="5px")
        self._warp_tile_shape.prop(padding_bottom="5px")
        self._access_A = mui.Input("Input A Access",
                                   init="8").prop(mui_margin="dense",
                                                  type="number")
        self._access_B = mui.Input("Input B Access",
                                   init="8").prop(mui_margin="dense",
                                                  type="number")

        name_to_top = {v.short_repr(): v for v in ALL_TENSOR_OP_MAP.values()}

        self.status = mui.Typography("").prop(
            font_family=_TERMINAL_FONT_FAMILY, white_space="pre-line")
        self.name_to_top = name_to_top
        self._tensorop_select = mui.Select("TensorOp",
                                           [(k, k)
                                            for k in name_to_top.keys()])
        self._tensorop_select.prop(mui_margin="dense")
        self.onchange_callbacks: List[Callable[[], Coroutine]] = []
        self.ee = pyee.AsyncIOEventEmitter()
        self.prev_values: Dict[str, mui.ValueType] = {}
        self.manual_reflower = three.FlexManualReflow()

        # self.active_designer = DesignerTypeV2.InputExtentA
        self.add_layout(self.get_layout())
        self.kernel = self._get_gemm_kernel()

    def _get_gemm_kernel(self):
        dtype_str = f"{self.dtype_ab.shortcut()},{self.dtype_ab.shortcut()},"
        dtype_str += f"{self.dtype_acc.shortcut()},{self.dtype_comp.shortcut()},"
        dtype_str += f"{self.dtype_c.shortcut()}"
        top = None 
        if self.tensor_op is not None:
            top = self.tensor_op.get_tensor_op()
        return GemmAlgoParams((self.tile_shape[0], self.tile_shape[1], self.tile_shape[2]), 
            (self.tile_shape[0], self.tile_shape[1], self.tile_shape[2]), 
            2, dtype_str, self.trans_a, self.trans_b, False, kernel.GemmAlgo.Turing, top)


    def get_layout(self):
        return {
            self.DtypeAB: self._dtypes_ab,
            self.DtypeAcc: self._dtypes_acc,
            self.DtypeComp: self._dtypes_comp,
            self.DtypeC: self._dtypes_c,
            self.TileShape: self._tile_shape,
            self.WarpTileShape: self._warp_tile_shape,
            self.TransA: self._trans_a,
            self.TransB: self._trans_b,
            self.LdsmA: self._ldsm_a,
            self.LdsmB: self._ldsm_b,

            self.AccessA: self._access_A,
            self.AccessB: self._access_B,
            self.TensorOp: self._tensorop_select,
        }

    def get_persist_props(self) -> Optional[Dict[str, Any]]:
        return {
            "params": self._get_control_values(),
        }

    async def set_persist_props_async(self, state: Dict[str, Any]) -> None:
        self.prev_values = self._get_control_values()
        if "params" in state:
            self.prev_values.update(state["params"])
            self.ee.emit(self.OnInputParamAChanged, self.prev_values, True)
            self.ee.emit(self.OnInputParamBChanged, self.prev_values, True)
            self.ee.emit(self.OnWarpNumChanged, self.prev_values, True)

            self.ee.emit(self.OnTensorOpChanged, self.prev_values, True)
            self.ee.emit(self.OnAnyChanged, self.prev_values, True)

    def get_changed_value(self):
        cur = self._get_control_values()
        changed = {}
        if self.prev_values:
            for k, v in cur.items():
                prev_v = self.prev_values[k]
                if v != prev_v:
                    changed[k] = v
        else:
            self.prev_values = cur
            changed = cur
        return changed

    def _get_control_values(self):
        value_dict = {}
        for k, v in self.get_layout().items():
            value_dict[k] = mui.get_control_value(v)
        return value_dict

    async def set_params(self):
        self.check_valid()
        changed = self.get_changed_value()
        inpa_changed = [
            self.DtypeAB, self.TileShape, self.TransA, self.AccessA
        ]
        inpb_changed = [
            self.DtypeAB, self.TileShape, self.TransB, self.AccessB
        ]
        num_warp_changed = [self.TileShape, self.WarpTileShape]
        self.kernel = self._get_gemm_kernel()
        if any(x in changed for x in inpa_changed):
            self.ee.emit(self.OnInputParamAChanged, changed, False)
        if any(x in changed for x in inpb_changed):
            self.ee.emit(self.OnInputParamBChanged, changed, False)
        if any(x in changed for x in num_warp_changed):
            self.ee.emit(self.OnWarpNumChanged, changed, False)

        if self.TensorOp in changed:
            self.ee.emit(self.OnTensorOpChanged, changed, False)
        self.ee.emit(self.OnAnyChanged, changed, False)

    def get_input_A_shape(self):
        res = [self.tile_shape[0], self.tile_shape[2]]
        if self.trans_a:
            res = res[::-1]
        return res

    def get_input_B_shape(self):
        res = [self.tile_shape[2], self.tile_shape[1]]
        if self.trans_b:
            res = res[::-1]
        return res

    def get_input_shape(self, is_A: bool):
        if is_A:
            res = [self.tile_shape[0], self.tile_shape[2]]
            if self.trans_a:
                res = res[::-1]
        else:
            res = [self.tile_shape[2], self.tile_shape[1]]
            if self.trans_b:
                res = res[::-1]
        return res

    def get_smem_shape(self, is_A: bool):
        if is_A:
            return [self.tile_shape[2], self.tile_shape[0]]
        else:
            return [self.tile_shape[2], self.tile_shape[1]]
    
    def get_input_access(self, is_A: bool):
        if is_A:
            return int(self._access_A.value)
        else:
            return int(self._access_B.value)

    def valid(self):
        try:
            self.check_valid()
            return True
        except:
            traceback.print_exc()
            return False

    @property 
    def dtype_ab(self):
        return dtypes.TVDTYPE_TO_DTYPE[int(self._dtypes_ab.value)]

    @property 
    def dtype_c(self):
        return dtypes.TVDTYPE_TO_DTYPE[int(self._dtypes_c.value)]

    @property 
    def dtype_acc(self):
        return dtypes.TVDTYPE_TO_DTYPE[int(self._dtypes_acc.value)]

    @property 
    def dtype_comp(self):
        return dtypes.TVDTYPE_TO_DTYPE[int(self._dtypes_comp.value)]

    def check_valid(self):
        # TODO more and more validation,
        # include ldmatrix
        ts = metaseq(*self._tile_shape.json())
        wts = metaseq(*self._warp_tile_shape.json())
        assert len(ts) == 3 and len(wts) == 3
        wc = ts // wts
        inp_A = self.get_input_A_shape()
        inp_B = self.get_input_B_shape()

        assert inp_A[0] * inp_A[1] * self.dtype_ab.bitsize() >= 128 * 8
        assert inp_B[0] * inp_B[1] * self.dtype_ab.bitsize() >= 128 * 8


    def get_input_iters(self, is_A: bool):
        if is_A:
            shape = self.get_input_A_shape()
            access = self.input_access_size_a
        else:
            shape = self.get_input_B_shape()
            access = self.input_access_size_b

        return shape[0] * shape[1] // access // 32 // self.num_warps

    @property
    def tile_shape(self):
        return metaseq(*self._tile_shape.json())

    @property
    def warp_tile_shape(self):
        return metaseq(*self._warp_tile_shape.json())

    @property
    def num_warps(self):
        wc = self.tile_shape // self.warp_tile_shape
        return wc.prod()

    @property
    def input_access_size_a(self):
        return int(self._access_A.value)

    @property
    def input_access_size_b(self):
        return int(self._access_B.value)

    @property
    def num_warp_k_iters(self):
        if self.tensor_op is not None:
            numk = self.warp_tile_shape[2] // self.tensor_op.shape[2]
        else:
            numk = self.warp_tile_shape[2]
        return numk

    @property
    def trans_a(self):
        return self._trans_a.checked

    @property
    def trans_b(self):
        return self._trans_b.checked

    @property
    def trans_c(self):
        return False

    @property
    def tensor_op(self):
        top = None
        if self._tensorop_select.value in self.name_to_top:
            top = self.name_to_top[self._tensorop_select.value]
        return top

class DesignerTypeV2(enum.Enum):
    InputExtentA = "InputExtentA"
    InputExtentB = "InputExtentB"
    SmemWriterA = "SmemWriterA"
    SmemWriterB = "SmemWriterB"


class GemmStatus(mui.FlexBox):
    def __init__(self, params: GemmParams):
        super().__init__()
        self.status = mui.Typography("").prop(
            font_family=_TERMINAL_FONT_FAMILY, white_space="pre-line")

        self.params = params

        params.ee.on(params.OnAnyChanged, self.on_param_change)
        self.add_layout({
            "s": self.status
        })

    async def on_param_change(self, changed, is_init: bool):
        await self.status.write(f"""
num warp: {self.params.num_warps}
k iterations: {self.params.num_warp_k_iters}
A input iters: {self.params.get_input_iters(True)}
B input iters: {self.params.get_input_iters(False)}

        """.strip())

class Errors(mui.FlexBox):
    def __init__(self):
        super().__init__()
        self.status = mui.Typography("").prop(
            font_family=_TERMINAL_FONT_FAMILY, white_space="pre-line")

        self.add_layout({

        })

    async def set_error(self, type: DesignerTypeV2, mode: str, error: str):
        name = f"{type.value}-{mode}"
        await self.update_childs({
            name: mui.Alert(error, "error", name).prop(margin="5px"),
        })

    async def remove_error(self, type: DesignerTypeV2, mode: str):
        name = f"{type.value}-{mode}"
        await self.remove_childs_by_keys([name])


@dataclasses.dataclass
class Extent:
    extent: List[int]
    label: str
    color: str

def _get_box2d(limit: int):
    res = three.Boxes2D(limit)
    res.prop(color="royalblue",
            line_color="black",
            alpha=0.0,
            line_width=0.5,
            hover_line_color="blue",
            hover_line_width=2)
    return res 

class GemmMatrix(three.ItemBox):
    """Gemm Matrix: component that support rich gemm data display.
    element data: {
        hover meta
    }
    lane hover data: {
        hover line width
    }
    warp hover data: {
        hover line width
    }

    lane selection color matrix: [shape[0] * shape[1], 4]
    lane selection index matrix: [shape[0] * shape[1]]

    # lane will override warp colors and merge warp metas

    """
    def __init__(self, warp_panel: WarpPanel, dtype: dtypes.DType, label: str, limit: int, enable_bank_view: bool = False) -> None:
        super().__init__({})
        self.dtype = dtype
        self.props.flex_direction = "column"
        self.props.margin = 0.5
        self.matrix = _get_box2d(limit)
        self.hover_matrix = _get_box2d(limit)
        self.extent_layouts = three.Group({})
        self.access_view = three.ToggleButton("AccessView", 3, 1, self.on_change_access_view)
        self.bank_view = three.ToggleButton("BankView", 3, 1, self.on_change_access_view)

        self.label = three.Text(label).prop(color="blue", font_size=1)
        self.shape = [0, 0]
        self.access = 1
        self.shape_prod = 0
        self.scale = 0.6
        self.extents: Dict[str, Extent] = {}
        self.enable_bank_view = enable_bank_view

        self.lane_select_color = np.zeros((0, 4), np.float32)
        self.lane_select_index = np.zeros((0,), np.int32)

        self.warp_panel = warp_panel

        layout: Dict[str, mui.Component] = {
            "l": three.FlexItem(self.label).prop(center_anchor=True),
            "a": three.FlexItem(self.access_view).prop(center_anchor=True, margin=0.1),
        }
        if enable_bank_view:
            layout["b"] = three.FlexItem(self.bank_view).prop(center_anchor=True, margin=0.1)
            # TODO when enable_bank_view, bank split lines should shown in matrix.
        layout["m"] = three.ItemBox({
            "m2": self.matrix,
            "e2": self.extent_layouts,
            "h2": self.hover_matrix,
        })
        self.add_layout(layout)
        self.warp_panel.ee.on(WarpPanel.LaneSelect, self.on_lane_select)
        warp_panel.ee.on(WarpPanel.Reset, self.on_reset)

    async def on_reset(self):
        if not isinstance(self.matrix.props.colors, mui.Undefined):
            self.matrix.props.colors[:] = 0
            await self.matrix.update_boxes(colors=self.matrix.props.colors)

    async def on_lane_select(self, enable, index):
        await self.on_lane_selects(enable, [index])

    async def on_lane_selects(self, enable, indexes: List[int]):
        if self.lane_select_index.size > 0:
            lane_colors = self.warp_panel.get_lane_colors_rgba(0.8)
            for index in indexes:
                valid = self.lane_select_index == index
                if enable:
                    self.lane_select_color[valid] = lane_colors[self.lane_select_index[valid]]
                else:
                    self.lane_select_color[valid] = 0
            access = self.access
            if not self.access_view.toggled:
                access = 1

            lane_select_color = self.lane_select_color[::access]
            # lane_select_color = self.lane_select_color

            await self.matrix.update_boxes(colors=lane_select_color)


    async def set_extents(self, extents: Dict[str, Extent]):
        self.extents = extents
        ext_layout: Dict[str, three.ArrowXYMeasure] = {}
        for name, ex in extents.items():
            exn = np.array(ex.extent)
            if self.access_view.toggled:
                exn[1] //= self.access
                exn[3] //= self.access
            exn = exn * self.scale
            # exn: [r, c, r, c]
            width_for_vis = exn[3] - exn[1]
            exn = exn[[1, 0, 3, 2]]
            exn[1] = -exn[1]
            exn[3] = -exn[3]
            label_size = min(self.scale * 4, width_for_vis * 0.7)
            arr = three.ArrowXYMeasure((exn[0], exn[1]),
                                        (exn[2], exn[3]),
                                        ex.label,
                                        label_size,
                                        self.scale * 0.75,
                                        self.scale,
                                        opacity=0.4,
                                        color=ex.color)
            ext_layout[name] = arr
        await self.extent_layouts.set_new_layout({**ext_layout})

    async def update_matrix(self, dtype: dtypes.DType, shape: List[int], access: int):
        self.shape = shape.copy()
        self.dtype = dtype
        self.access = access
        if not self.access_view.toggled:
            access = 1
        shape_prod_raw = shape[0] * shape[1]
        if self.lane_select_index.size != shape_prod_raw:
            self.lane_select_index = np.full([shape_prod_raw], -1, np.int32)
            self.lane_select_color = np.zeros([shape_prod_raw, 4], np.float32)
        
        # TODO if smem is padded instead of shuffled...
        if self.bank_view.toggled:
            num_bank = shape_prod_raw * self.dtype.bitsize() // SMEM_BANK_SIZE_BITS
            bank_element = SMEM_BANK_SIZE_BITS // self.dtype.bitsize()
            shape = [num_bank, bank_element]
        shape[1] //= access
        shape_prod = shape[0] * shape[1]
        assert shape_prod <= self.matrix.limit
        dimensions = np.ones((1, ), np.float32) * self.scale
        centers = _create_matrix_centers(shape)
        lane_select_color = self.lane_select_color[::access]

        colors = lane_select_color
        # valid = self.lane_select_index != -1
        # self.lane_select_color[:] = 0
        # self.lane_select_color[valid] = lane_colors[self.lane_select_index[valid]]
        await self.matrix.update_boxes(centers * self.scale,
                                       dimensions,
                                       colors=colors)

    async def update_lane_index_matrix(self, lane_index: np.ndarray):
        assert self.lane_select_index.size == lane_index.size, f"size mismatch, {self.lane_select_index.size}, {lane_index.size}"
        self.lane_select_index = lane_index

    async def on_change_access_view(self, enable: bool):
        await self.update_matrix(self.dtype, self.shape, self.access)
        if self.extents:
            await self.set_extents(self.extents)

    async def on_change_bank_view(self, enable: bool):
        await self.update_matrix(self.dtype, self.shape, self.access)
        if enable:
            await self.set_extents({}) # extent must be disabled if bank view
        else:
            if self.extents:
                await self.set_extents(self.extents)

class TensorOpViewer(three.Group):

    def __init__(self, params: GemmParams, warp_panel: WarpPanel) -> None:
        super().__init__({})
        self.params = params
        self.warp_panel = warp_panel

        self.operandA = _get_box2d(10000)
        self.operandB = _get_box2d(10000)
        self.operandC = _get_box2d(10000)
        # self.html.prop(position=(-4, -6, 0), transform=True)
        self.show_lines = three.Segments(1000, 2, color="red")
        self.add_layout({
            "matrix":
            three.Flex({
                "Inputs":
                three.HBox({
                    "AText":
                    three.FlexItem(
                        three.Text("A").prop(color="blue",
                                             font_size=1)).prop(margin=0.1, center_anchor=True),
                    "A":
                    three.FlexItem(self.operandA).prop(margin=0.1),
                    "BText":
                    three.FlexItem(three.Text("B").prop(color="blue", font_size=1)).prop(margin=0.1, center_anchor=True),
                    "B":
                    three.FlexItem(self.operandB).prop(margin=0.1),
                }).prop(padding=0.2),
                "Outputs":
                three.HBox({
                    "C":
                    three.FlexItem(self.operandC).prop(margin=0.1),
                }).prop(padding=0.2),
                "reflow":
                three.FlexAutoReflow(),
            }).prop(flex_direction="column", position=(0, 6, 0)),
            # "html0":
            # self.html,
            # "wraps":
            # self.warp_panel.prop(position=(-8, 0, 0)),
        })
        params.ee.on(GemmParams.OnTensorOpChanged, self.on_change)
        warp_panel.ee.on(WarpPanel.LaneSelect, self.on_warp_select)
        warp_panel.ee.on(WarpPanel.Reset, self.on_reset)

    async def on_warp_select(self, enable, index):
        if self.params.tensor_op is not None:
            top = self.params.tensor_op
            map_a = top.a_map()  # [Nfrag, 32, 2]
            map_b = top.b_map()
            map_c = top.c_map()
            a_coord = map_a[:, index]  # [Nfrag, 2]
            b_coord = map_b[:, index]  # [Nfrag, 2]
            c_coord = map_c[:, index]  # [Nfrag, 2]
            if enable:
                click_color = self.warp_panel.warp_btns[
                    index].mesh.props.click_color
                if not isinstance(click_color, mui.Undefined):
                    color = (*mui.colors.str_to_rgb_float(click_color), 0.8)
                else:
                    color = (0.5, 0, 0, 0.8)
            else:
                color = (0, 0, 0, 0.0)
            self.operandA.props.colors.reshape(*top.shape_a,
                                               4)[a_coord[:, 0],
                                                  a_coord[:, 1]] = color
            self.operandB.props.colors.reshape(*top.shape_b,
                                               4)[b_coord[:, 0],
                                                  b_coord[:, 1]] = color
            self.operandC.props.colors.reshape(*top.shape_c,
                                               4)[c_coord[:, 0],
                                                  c_coord[:, 1]] = color
            await self.operandA.update_boxes(colors=self.operandA.props.colors)
            await self.operandB.update_boxes(colors=self.operandB.props.colors)
            await self.operandC.update_boxes(colors=self.operandC.props.colors)

    async def on_reset(self):
        if not isinstance(self.operandA.props.colors, mui.Undefined):
            self.operandA.props.colors[:, :3] = 0
            self.operandA.props.colors[:, 3] = 0.0
            await self.operandA.update_boxes(colors=self.operandA.props.colors)
        if not isinstance(self.operandB.props.colors, mui.Undefined):
            self.operandB.props.colors[:, :3] = 0
            self.operandB.props.colors[:, 3] = 0.0
            await self.operandB.update_boxes(colors=self.operandB.props.colors)
        if not isinstance(self.operandC.props.colors, mui.Undefined):
            self.operandC.props.colors[:, :3] = 0
            self.operandC.props.colors[:, 3] = 0.0
            await self.operandC.update_boxes(colors=self.operandC.props.colors)

    async def on_change(self, changed: Dict[str, Any], is_init: bool):
        top = self.params.tensor_op
        if top is not None:
            scale = 0.6
            a_shape = [top.shape[0], top.shape[2]]
            b_shape = [top.shape[2], top.shape[1]]
            c_shape = [top.shape[0], top.shape[1]]
            centersA = _create_matrix_centers(a_shape)
            centersB = _create_matrix_centers(b_shape)
            centersC = _create_matrix_centers(c_shape)
            dimensions = np.ones((1, ), np.float32) * scale
            colorsA = np.zeros((top.km, 4), np.float32)
            colorsB = np.zeros((top.kn, 4), np.float32)
            colorsC = np.zeros((top.mn, 4), np.float32)
            colorsA[:, -1] = 0.0
            colorsB[:, -1] = 0.0
            colorsC[:, -1] = 0.0
            await self.operandA.update_boxes(centersA * scale,
                                             dimensions,
                                             colors=colorsA)
            await self.operandB.update_boxes(centersB * scale,
                                             dimensions,
                                             colors=colorsB)
            await self.operandC.update_boxes(centersC * scale,
                                             dimensions,
                                             colors=colorsC)

    def is_visible_undefined(self):
        return isinstance(self.props.visible, mui.Undefined)

    async def set_visible(self, visible: bool):
        await self.update_object3d(visible=visible)
        # await self.html.update_object3d(visible=visible)

class GemmCanvas(three.ItemBox):
    """
              TopViewer
    WarpPanel A     B 
              SmemA SmemB

    """
    FuncWarpExtent = "input_warp_extent_map"

    def __init__(self, params: GemmParams, warp_panel: "WarpPanel"):
        self.input_mat_A = GemmMatrix(warp_panel, params.dtype_ab, "A", 10000)
        self.input_mat_B = GemmMatrix(warp_panel, params.dtype_ab, "B", 10000)
        # TODO number of smem elements may overflow?
        self.smem_mat_A = GemmMatrix(warp_panel, params.dtype_ab, "SmemA", 20000, enable_bank_view=True)
        self.smem_mat_B = GemmMatrix(warp_panel, params.dtype_ab, "SmemB", 20000, enable_bank_view=True)


        self.params = params
        self.warp_panel = warp_panel
        self.mode = DesignerTypeV2.InputExtentA

        super().__init__({**self._get_layout_by_mode(self.mode)})
        self.props.flex_direction = "column"
        self.props.width = "100%"
        self.props.height = "100%"

        if self.mode == DesignerTypeV2.SmemWriterA:
            if self.params.trans_a:
                self.props.flex_direction = "column"
            else:
                self.props.flex_direction = "row"
        if self.mode == DesignerTypeV2.SmemWriterB:
            if self.params.trans_b:
                self.props.flex_direction = "row"
            else:
                self.props.flex_direction = "column"


    def _get_layout_by_mode(self, mode: DesignerTypeV2):
        if mode == DesignerTypeV2.InputExtentA:
            return {
                "A": self.input_mat_A,
            }
        if mode == DesignerTypeV2.InputExtentB:
            return {
                "B": self.input_mat_B,
            }
        if mode == DesignerTypeV2.SmemWriterA:
            return {
                "A": self.input_mat_A,
                "SmemA":self.smem_mat_A,
            }
        if mode == DesignerTypeV2.SmemWriterB:
            return {
                "B": self.input_mat_B,
                "SmemB": self.smem_mat_B,
            }
        return {}

    async def set_mode(self, mode: DesignerTypeV2):
        self.mode = mode
        if mode == DesignerTypeV2.SmemWriterA:
            if self.params.trans_a:
                await self.send_and_wait(self.update_event(flex_direction="column"))
            else:
                await self.send_and_wait(self.update_event(flex_direction="row"))
        if mode == DesignerTypeV2.SmemWriterB:
            if self.params.trans_b:
                await self.send_and_wait(self.update_event(flex_direction="row"))
            else:
                await self.send_and_wait(self.update_event(flex_direction="column"))

        await self.set_new_layout({**self._get_layout_by_mode(mode)})



class Designer(three.Group):

    def __init__(self, type: DesignerTypeV2, params: GemmParams,
                 warp_panel: WarpPanel, editor: flowapp.AppEditor,
                 canvas: GemmCanvas) -> None:
        super().__init__({})
        self.type = type
        self.params = params
        self.warp_panel = warp_panel
        self.editor = editor
        self.canvas = canvas

    @property
    def code(self) -> str:
        # ensure we access code from global storage.
        return self.code_dict["code"]

    @property
    def code_key(self) -> str:
        # ensure we access code from global storage.
        return self.code_dict["code_key"]

    @property
    def code_dict(self) -> Dict[str, str]:
        # ensure we access code from global storage.
        code_store = flowapp.get_app_storage()["code"]
        if self.type.value not in code_store:
            code_store[self.type.value] = {
                "code": "",
                "code_key": "",
            }
        return code_store[self.type.value]

    async def sync_code(self):
        await self.editor.set_editor_value(self.code, "python")

    async def save_code(self, code: str):
        pass

class SmemWriteDesigner(Designer):
    FuncInputRead = "input_read"
    FuncSmemWrite = "smem_write"
    def __init__(self, is_A: bool, params: GemmParams, warp_panel: WarpPanel,
                 editor: flowapp.AppEditor, canvas: GemmCanvas) -> None:
        if is_A:
            t = DesignerTypeV2.SmemWriterA
        else:
            t = DesignerTypeV2.SmemWriterB
        super().__init__(t, params, warp_panel, editor, canvas)
        if is_A:
            self.gemm_mat = self.canvas.smem_mat_A
            self.input_gemm_mat = self.canvas.input_mat_A

            params.ee.on(GemmParams.OnInputParamAChanged, self.on_change)
        else:
            self.gemm_mat = self.canvas.smem_mat_B
            self.input_gemm_mat = self.canvas.input_mat_B

            params.ee.on(GemmParams.OnInputParamBChanged, self.on_change)
        self.is_A = is_A

    def get_code_key(self):
        return f"{self.params.num_warps}_{self.params.num_warp_k_iters}"


    async def on_change(self, changed: Dict[str, Any], is_init: bool):
        self.params.check_valid()
        shape = self.params.get_smem_shape(self.is_A)
        inp_shape = self.params.get_input_shape(self.is_A)
        access = self.params.get_input_access(self.is_A)
        await self.gemm_mat.update_matrix(self.params.dtype_ab, shape, access)
        if self.get_code_key() != self.code_key:
            # we need to reload code here.
            # print("CHANGED", self.code_key)
            code = f"""
def {self.FuncInputRead}(warp: int, lane: int, i: int):
    # Input ({inp_shape}) -> Smem ({shape}): Input Part.
    # return access idx, NOT elem idx, or None if not sure.
    return None\n

def {self.FuncSmemWrite}(warp: int, lane: int, i: int):
    # Input ({inp_shape}) -> Smem ({shape}): Smem Part.
    # two constraints:
    # 1. smem writes must not access same bank
    #    in a group (8-128bit, 16-64bit, 32-32bit)
    # 2. warp reads must not access same bank 
    # we will follow this to detect smem read/warp write
    # bank conflicts when you save code.
    # return access idx, NOT elem idx, or None if not sure.    
    return None\n"""
            flowapp.get_app_storage()["code"][self.type.value] = {
                "code": code,
                "code_key": self.get_code_key(),
            }
            if self.type == self.canvas.mode:
                await self.sync_code()
        if is_init:
            if self.type == self.canvas.mode:
                await self.sync_code()
        await self._run_code(self.code)

    async def _run_code(self, code: str):
        mod = {}
        exec(code, mod)
        func = mod[self.FuncSmemWrite]
        func_input_read = mod[self.FuncInputRead]

        iters = self.params.get_input_iters(self.is_A)
        num_access = self.params.get_input_access(self.is_A)
        select_index = np.full([32 * self.params.num_warps * iters * num_access], -1, np.int32)
        input_select_index = np.full([32 * self.params.num_warps * iters * num_access], -1, np.int32)
        # TODO if out of range...
        for w in range(self.params.num_warps):
            for l in range(32):
                for i in range(iters):
                    input_read_idx = func_input_read(w, l, i)
                    smem_write_idx = func(w, l, i)
                    if smem_write_idx is not None:
                        select_index[smem_write_idx * num_access:(smem_write_idx + 1) * num_access] = l
                    if input_read_idx is not None:
                        input_select_index[input_read_idx * num_access:(input_read_idx + 1) * num_access] = l
        await self.gemm_mat.update_lane_index_matrix(select_index)
        await self.input_gemm_mat.update_lane_index_matrix(input_select_index)

    async def save_code(self, code: str):
        await self._run_code(code)
        flowapp.get_app_storage()["code"][self.type.value] = {
            "code": code,
            "code_key": self.get_code_key(),
        }

class InputExtentDesigner(Designer):
    FuncWarpExtent = "input_warp_extent_map"
    def __init__(self, is_A: bool, params: GemmParams, warp_panel: WarpPanel,
                 editor: flowapp.AppEditor, canvas: GemmCanvas) -> None:
        if is_A:
            t = DesignerTypeV2.InputExtentA
        else:
            t = DesignerTypeV2.InputExtentB
        super().__init__(t, params, warp_panel, editor, canvas)
        if is_A:
            self.gemm_mat = self.canvas.input_mat_A
            params.ee.on(GemmParams.OnInputParamAChanged, self.on_change)
        else:
            self.gemm_mat = self.canvas.input_mat_B

            params.ee.on(GemmParams.OnInputParamBChanged, self.on_change)
        self.is_A = is_A

    async def on_change(self, changed: Dict[str, Any], is_init: bool):
        self.params.check_valid()
        shape = self.params.get_input_shape(self.is_A)
        access = self.params.get_input_access(self.is_A)
        await self.gemm_mat.update_matrix(self.params.dtype_ab, shape, access)
        code_key = f"{self.params.num_warps}"
        if code_key != self.code_key:
            # we need to reload code here.
            # print("CHANGED", self.code_key)
            code = f"""
def {self.FuncWarpExtent}(warp: int):
    # return [start_row, start_col, end_row, end_col]
    return None
            """
            flowapp.get_app_storage()["code"][self.type.value] = {
                "code": code,
                "code_key": code_key,
            }
            if self.type == self.canvas.mode:
                await self.sync_code()
        if is_init:
            if self.type == self.canvas.mode:
                await self.sync_code()

        print("ON CHANGE")
        await self._run_code(self.code)

    async def _run_code(self, code: str):
        mod = {}
        exec(code, mod)
        extents: Dict[str, Extent] = {}
        func = mod[self.FuncWarpExtent]
        boxes = []
        box_warp_idxes = []
        
        for i in range(self.params.num_warps):
            ex = func(i)
            if ex is not None:
                exn = np.array(ex)
                boxes.append(exn)
                box_warp_idxes.append(i)
                extents[f"{i}"] = Extent(ex, f"W{i}", "black")
        box_warp_idxes_np = np.array(box_warp_idxes)
        self_overlap = utils.boxes_self_overlap(np.array(boxes))
        box_wrong = set([*box_warp_idxes_np[self_overlap[0]], *box_warp_idxes_np[self_overlap[1]]])
        if box_wrong:
            await self.params.errors.set_error(self.type, "Regular", "Extent Overlap!!!")
        else:
            await self.params.errors.remove_error(self.type, "Regular")
        for wrong_idx in box_wrong:
            extents[str(wrong_idx)].color = "red"
        await self.gemm_mat.set_extents(extents)

    async def save_code(self, code: str):
        # [16 * warp, 0, (warp + 1) * 16, 32]
        # [0, warp * 16, 32, (warp + 1) * 16]
        # print("SAVE", self.type, id(flowapp.get_app_storage()))
        await self._run_code(code)
        flowapp.get_app_storage()["code"][self.type.value] = {
            "code": code,
            "code_key": f"{self.params.num_warps}",
        }


class TaskItem(mui.FlexBox):

    def __init__(self, desp: str, on_click: Callable[[], Coroutine]) -> None:
        super().__init__()
        self.desp = mui.Typography(desp).prop(variant="h3")
        self.btn = mui.Button("Enter", on_click).prop(mui_color="primary")
        self.on_click = on_click
        self.props.align_items = "center"
        self.add_layout({
            "divider": mui.HDivider(),
            "desp": self.desp,
            "btn": self.btn,
        })

    async def set_finish(self):
        await self.send_and_wait(
            self.btn.update_event(name="Done", mui_color="success"))

    async def set_working(self):
        await self.send_and_wait(
            self.btn.update_event(name="Enter", mui_color="primary"))


class GemmDesigner(flowapp.EditableLayoutApp):
    TopViewer = "TopViewer"
    InputIterA = "InputIterA"
    InputIterB = "InputIterB"

    def __init__(self) -> None:
        super().__init__(False)
        self.root.props.min_height = 0
        self.root.props.min_width = 0
        self.set_init_window_size([800, 600])
        self.get_persist_storage()["code"] = {}
        self.code_storage: Dict[str, Dict[str, str]] = self.get_persist_storage()["code"]
        for k in DesignerTypeV2:
            self.code_storage[k.value] = {
                "code": "",
                "code_key": ""
            }
        print(self.get_persist_storage())

    async def app_initialize_async(self):
        pass

    def app_create_layout(self) -> Dict[str, mui.MUIComponentType]:
        cam = three.OrthographicCamera(True, near=0.1, far=1000, zoom=50.0)
        cam.prop(position=(0, 0, 10), up=(0, 0, 1))
        ctrl = three.MapControl()
        ctrl.props.enable_rotate = False
        self.errors = Errors()
        self.gemm_params = GemmParams(self.errors)
        self.warp_panel = WarpPanel(0, 1)
        self.warp_panel.props.position = (-10, 0, 0)
        self.gemm_canvas = GemmCanvas(self.gemm_params, self.warp_panel)
        self.gemm_params.ee.on(GemmParams.OnWarpNumChanged,
                               self.on_warp_num_change)
        self.gemm_status = GemmStatus(self.gemm_params)
        self.top_viewer = TensorOpViewer(self.gemm_params, self.warp_panel)
        
        self.iterA = InputExtentDesigner(True, self.gemm_params, self.warp_panel,
                                   self.code_editor, self.gemm_canvas)
        self.iterB = InputExtentDesigner(False, self.gemm_params, self.warp_panel,
                                   self.code_editor, self.gemm_canvas)
        self.smem_writer_A = SmemWriteDesigner(True, self.gemm_params, self.warp_panel,
                                   self.code_editor, self.gemm_canvas)
        self.smem_writer_B = SmemWriteDesigner(False, self.gemm_params, self.warp_panel,
                                   self.code_editor, self.gemm_canvas)

        self.designers: Dict[str, Designer] = {
            DesignerTypeV2.InputExtentA.value: self.iterA,
            DesignerTypeV2.InputExtentB.value: self.iterB,
            DesignerTypeV2.SmemWriterA.value: self.smem_writer_A,
            DesignerTypeV2.SmemWriterB.value: self.smem_writer_B,
        }
        self.canvas = three.ThreeCanvas({
            "cam": cam,
            "ctrl": ctrl,
            "warp_panel": self.warp_panel,
            "canvas": three.Flex({
                "c": self.gemm_canvas,
                "reflow":three.FlexAutoReflow(),
                # "mreflow3": self.gemm_params.manual_reflower,
            }).prop(position=(0, 0, 0), flex_direction="column", size=(1000, 1000, 0)),
            "designer": self.iterA,
            "top": self.top_viewer.prop(position=(0, 15, 0)),

            # "ax": three.AxesHelper(10),
            # "arrow": three.ArrowXYMeasure((0, 0), (10, 10), "WTF", 1, 0.5, 1, 0.4)
            # "itera": self.iterA,
            # "iterb": self.iterB,
            # "tr": three.TransformControls().prop(object3d_uid="root.d3v.d3.top_viewer")
        })
        return {
            "d3v":
            mui.VBox({
                "d3":
                self.canvas,
                "status":
                mui.HBox({
                    "p": self.gemm_status.prop(padding="5px"),
                }).prop(position="absolute",
                        top="40%",
                        left="0",
                        pointer_events="none",
                        justify_content="flex-start"),

                "params":
                mui.HBox({
                    # "p": self.gemm_status.prop(padding="5px"),
                    "p2": mui.Accordion(
                        mui.AccordionSummary({
                            "a": mui.Typography("Gemm Params")
                        }),
                        mui.AccordionDetails({
                            "a": mui.VBox({
                                "params": self.gemm_params.prop(flex_flow="column"),
                                # "update": mui.Button("Box2d", self.on_box2d_update),
                                "sync": mui.Button("Set Gemm Params", self._set_gemm_params),
                            })
                        })
                    ),
                }).prop(position="absolute",
                        top=0,
                        right=0,
                        z_index=5,
                        justify_content="flex-end"),
                "errrors": self.errors.prop(position="absolute",
                        flex_flow="column",
                        bottom=0,
                        left=0,
                        z_index=5,
                        justify_content="flex-end",
                        margin="5px",
                        max_height="50%",
                        overflow_y="auto"),
                "tasks":
                mui.VBox({
                    "title":
                    mui.Typography(f"Tasks: {1}/{14}").prop(
                        variant="h2", align_self="center"),
                    # "update": mui.Button("Box2d", self.on_box2d_update),
                    "setInpARange":
                    TaskItem("Set A Warp Range", self._enter_view_iter_a),
                    "setInpBRange":
                    TaskItem("Set B Warp Range", self._enter_view_iter_b),
                    "smemwriteA":
                    TaskItem("Smem Write A", self._enter_smem_writer_a),
                    "smemwriteB":
                    TaskItem("Smem Write B", self._enter_smem_writer_b),
                }).prop(position="absolute",
                        top=0,
                        left=0,
                        z_index=5,
                        justify_content="flex-start",
                        margin="5px",
                        border="solid gray 2px",
                        background_color="white",
                        max_height="50%",
                        overflow_y="auto")
            }).prop(position="relative", flex=1, min_height=0),
        }

    def get_active_designer(self):
        return self.designers[self.gemm_canvas.mode.value]

    async def select_designer(self, active_designer: DesignerTypeV2):
        assert active_designer.value in self.designers, str(active_designer)
        # self.gemm_params.active_designer = active_designer
        await self.gemm_canvas.set_mode(active_designer)
        # ev = flowapp.AppEvent("", {})
        # for k,v in self.designers.items():
        #     ev += v.update_event(visible=k == self.active_designer)
        # await self.canvas.update_childs(
        #     {"designer": self.designers[active_designer]})
        # await self.canvas.send_app_event_and_wait(ev)
        await self.code_editor.set_editor_value(
            self.designers[active_designer.value].code, "python")
        # await asyncio.sleep(0.5)
        # await self.gemm_params.manual_reflower.reflow()

    async def _enter_view_iter_a(self):
        await self.select_designer(DesignerTypeV2.InputExtentA)

    async def _enter_view_iter_b(self):
        await self.select_designer(DesignerTypeV2.InputExtentB)

    async def _enter_smem_writer_a(self):
        await self.select_designer(DesignerTypeV2.SmemWriterA)
    
    async def _enter_smem_writer_b(self):
        await self.select_designer(DesignerTypeV2.SmemWriterB)

    async def _enter_task(self):
        raise NotImplementedError

    async def on_warp_num_change(self, changed: Dict[str, Any], is_init: bool):
        await self.warp_panel.reset_num_warps(self.gemm_params.num_warps)

    async def _set_gemm_params(self):
        self.gemm_params.check_valid()
        await self.gemm_params.set_params()
        # self.gemm_params.ee.emit("changed")
        # await self.warp_panel.reset_num_warps(self.gemm_params.num_warps)

        # if self.gemm_params.valid():
        #     await self.top_viewer.on_tensorop_select(self.gemm_params._tensorop_select.value)

    async def _select_designer(self):
        if self.top_viewer.is_visible_undefined():
            await self.top_viewer.set_visible(False)
        else:
            await self.top_viewer.set_visible(not self.top_viewer.props.visible
                                              )

    async def handle_code_editor_event(self,
                                       event: flowapp.AppEditorFrontendEvent):
        if event.type == flowapp.AppEditorFrontendEventType.Save:
            await self.get_active_designer().save_code(event.data)

    # async def _restore_simple_app_state(self, state: Dict[str, Any]):
    #     await super()._restore_simple_app_state(state)
    #     self.gemm_params.prev_values = self.gemm_params._get_control_values()
    #     await self.iterA.on_change(self.gemm_params.prev_values)