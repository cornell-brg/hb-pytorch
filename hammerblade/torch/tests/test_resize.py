"""
Tensor Manipulations

Bandhav Veluri
"""

import torch

torch.manual_seed(42)

def test_resize_1():
    x1 = torch.rand(2, 3)
    x1_h = x1.hammerblade()

    x1.resize_(12)
    x1_h.resize_(12)

    x1_h_cpu = x1_h.cpu()

    assert x1_h.device == torch.device("hammerblade")
    assert x1_h.shape == x1.shape
    for i in range(6):
        assert torch.equal(x1_h_cpu[i], x1[i])

def test_resize_to_0():
    x1 = torch.rand(2, 3)
    x1_h = x1.hammerblade()

    x1.resize_(0)
    x1_h.resize_(0)

    assert x1_h.device == torch.device("hammerblade")
    assert x1_h.shape == x1.shape
