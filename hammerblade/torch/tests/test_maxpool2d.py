"""
Unit tests for maxpool2d operator
03/18/2020 Bandhav Veluri
"""

import torch
import torch.nn.functional as F

def test_max_pool2d_1():
    x = torch.rand(1, 1, 5, 5)
    x_hb = x.hammerblade()

    y = F.max_pool2d(x, 3, stride=2)
    y_hb = F.max_pool2d(x_hb, 3, stride=2)
    print(y)
    print(y_hb)

    assert torch.allclose(y, y_hb.cpu())

if __name__ == "__main__":
    test_max_pool2d_1()
