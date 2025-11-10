import numpy as np

from cumm import tensorview as tv


def test_tensor_basic():
    a = np.random.uniform(0, 1, size=[5, 5]).astype(np.float32)
    a_tv = tv.from_numpy(a).cuda()
    a2 = a_tv[:, :2]

    # a3 = tv.zeros([5, 2], tv.float32, -1)
    # a3.copy_2d_pitched_(a2)

    a3 = a2.cpu()
    # a_tv = tv.from_numpy(a).cuda()
    print(a)
    print(a3.numpy())

if __name__ == "__main__":
    test_tensor_basic()
