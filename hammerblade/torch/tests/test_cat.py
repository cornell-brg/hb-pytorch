"""
tests of simple cat kernel.
"""

import torch

def _test_torch_cat(x):
    x_h = x.hammerblade()
    y = torch.cat([x, x, x], 0)
    y_h = torch.cat([x_h, x_h, x_h], 0)
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y, y_h.cpu())

def test_cat_1():
    x = torch.ones(10)
    _test_torch_cat(x)

def test_cat_1_dif_sizes():
    x = torch.randn(3)
    y = torch.randn(2)
    z = torch.tensor([])
    x_h = x.hammerblade()
    y_h = y.hammerblade()
    z_h = z.hammerblade()
    a = torch.cat([x, y, z], 0)
    a_h = torch.cat([x_h, y_h, z_h], 0)
    assert a_h.device == torch.device("hammerblade")
    assert torch.allclose(a, a_h.cpu())

def test_cat_2():
    x = torch.randn(3, 4)
    _test_torch_cat(x)

def test_cat_3():
    x = torch.randn(3, 4, 5)
    _test_torch_cat(x)
