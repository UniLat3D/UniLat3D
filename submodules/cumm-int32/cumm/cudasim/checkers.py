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

from typing import List, Optional, Tuple, Union

import numpy as np

from cumm import cudasim
from cumm.core_cc.csrc.arrayref import ArrayPtr


async def smem_bank_conflicit_check(array_ptr: ArrayPtr,
                                    idx_access: int) -> None:
    # perform bank conflicit detection here.
    lane_id = cudasim.get_lane_id()
    # bank conflicit: 2 or more lane in a warp access
    # different 4B inside a same bank
    warp_data = await cudasim.warp_gather(
        (idx_access, array_ptr), 0)  # type: List[Tuple[int, ArrayPtr]]
    if lane_id == 0:
        b4_idx_to_idxes = ArrayPtr.check_smem_bank_conflicit(warp_data)
        if len(b4_idx_to_idxes) > 0:
            conflicit_indexes = sum(b4_idx_to_idxes.values(), [])
            msg = (
                f"BANK CONFLICIT!!! Pointers: {[warp_data[i][1] for i in conflicit_indexes]}, "
                f"Accesses: {[warp_data[i][0] for i in conflicit_indexes]}, "
                f"Indexes: {conflicit_indexes}")
            raise RuntimeError(msg)
    await cudasim.warp_wait()
