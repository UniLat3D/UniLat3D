import numpy as np
from cumm.utils.array_grad_check import check_array_op_grad
def test_uqmat_and_grad():
    # init cuda
    quat = np.random.uniform(-1, 1, size=[5, 4]).astype(np.float32)
    check_array_op_grad(quat, [3, 3], "uqmat_colmajor", "uqmat_colmajor_grad", delta=1e-4)

if __name__ == "__main__":
    test_uqmat_and_grad()