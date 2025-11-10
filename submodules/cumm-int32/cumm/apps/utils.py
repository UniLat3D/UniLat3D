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

import numpy as np 
import numba 

@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=0.0):
    """calculate box iou. note that jit version runs 2x faster than cython in 
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps) *
                        (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def boxes_in_box(boxes: np.ndarray, box: np.ndarray):
    boxes = boxes.astype(np.float32)
    box = box.astype(np.float32)

    assert boxes.ndim == 2 and box.ndim == 1
    iou = iou_jit(boxes, box[np.newaxis])
    return iou.reshape(-1) > 0

def boxes_self_overlap(boxes: np.ndarray):
    if boxes.nbytes == 0:
        boxes = boxes.reshape(0, 4)
    boxes = boxes.astype(np.float32)
    iou = iou_jit(boxes, boxes)
    ar = np.arange(boxes.shape[0])
    iou[ar, ar] = 0

    return np.nonzero(iou > 0)
