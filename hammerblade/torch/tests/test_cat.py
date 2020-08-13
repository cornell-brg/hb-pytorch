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

def test_cat_2():
    x = torch.randn(3, 4)
    _test_torch_cat(x)

def test_cat_3():
    x = torch.randn(3, 4, 5)
    _test_torch_cat(x)
