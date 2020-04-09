"""
Tensor Manipulations

Bandhav Veluri
"""

import torch
import random

torch.manual_seed(42)
random.seed(42)

def test_view_1():
    x1 = torch.rand(2, 3)
    x1_h = x1.hammerblade()

    x1 = x1.view(2 * 3)
    x1_h = x1_h.view(2 * 3)

    assert x1_h.device == torch.device("hammerblade")
    assert torch.equal(x1_h.cpu(), x1)

def test_view_2():
    x1 = torch.rand(2, 3)
    x1_h = x1.hammerblade()

    x1 = x1.view(1, 2 * 3)
    x1_h = x1_h.view(1, 2 * 3)

    assert x1_h.device == torch.device("hammerblade")
    assert torch.equal(x1_h.cpu(), x1)

def test_view_3():
    x1 = torch.rand(4, 6)
    x1_h = x1.hammerblade()

    x1 = x1.view(2, 3, 4)
    x1_h = x1_h.view(2, 3, 4)

    assert x1_h.device == torch.device("hammerblade")
    assert torch.equal(x1_h.cpu(), x1)

def test_view_4():
    x1 = torch.rand(2, 3)
    x2 = torch.rand(2, 3)
    x1_h = x1.hammerblade()
    x2_h = x2.hammerblade()

    x1_h = x1_h.view(3, 2)
    x2_h = x2_h.view(3, 2)

    z = x1 * x2
    z_h = x1_h * x2_h

    assert z_h.shape == (3, 2)
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z.view(3, 2))

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
