"""
tests of simple cat kernel.
"""
import torch
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

def _test_torch_cat(x, y, z, i):
    x_h = x.hammerblade()
    y_h = y.hammerblade()
    z_h = z.hammerblade()
    a = torch.cat([x, y, z], i)
    a_h = torch.cat([x_h, y_h, z_h], i)
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(a, a_h.cpu())

def _test_torch_cat2(x, y, i):
    x_h = x.hammerblade()
    y_h = y.hammerblade()
    a = torch.cat([x, y], i)
    a_h = torch.cat([x_h, y_h], i)
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(a, a_h.cpu())

def test_cat_1():
    x = torch.ones(10)
    _test_torch_cat(x, x, x, 0)

def test_cat_1_dif_sizes():
    x = torch.randn(3)
    y = torch.randn(2)
    z = torch.tensor([])
    _test_torch_cat(x, y, z, 0)

def test_cat_2_dim0():
    x = torch.randn(3, 4)
    _test_torch_cat(x, x, x, 0)

def test_cat_2_dim1():
    x = torch.randn(3, 4)
    _test_torch_cat(x, x, x, 0)

def test_cat_2_dif_sizes_dim0():
    x = torch.randn(3, 4)
    y = torch.randn(2, 4)
    z = torch.randn(4, 4)
    _test_torch_cat(x, y, z, 0)

def test_cat_2_dif_sizes_dim1():
    x = torch.randn(4, 3)
    y = torch.randn(4, 2)
    z = torch.randn(4, 4)
    _test_torch_cat(x, y, z, 1)

def test_cat_2_dif_sizes_dim2():
     x = torch.randn(4, 3)
     y = torch.randn(4, 2)
     z = torch.randn(4, 4)
     _test_torch_cat(x, y, z, -1)

def test_cat_3_dim0():
    x = torch.randn(3, 4, 5)
    _test_torch_cat(x, x, x, 0)

def test_cat_3_dim1():
    x = torch.randn(3, 4, 5)
    _test_torch_cat(x, x, x, 1)

def test_cat_3_dim2():
    x = torch.randn(3, 4, 5)
    _test_torch_cat(x, x, x, 2)

def test_cat_3_dif_sizes_dim0():
    x = torch.randn(3, 4, 5)
    y = torch.randn(2, 4, 5)
    z = torch.randn(4, 4, 5)
    _test_torch_cat(x, y, z, 0)

def test_cat_3_dif_sizes_dim0():
    x = torch.randn(256, 12, 64)
    y = torch.randn(256, 12, 64)
    _test_torch_cat2(x, y, 0)

def test_cat_3_dif_sizes_dim1():
    x = torch.randn(256, 12, 64)
    y = torch.randn(256, 12, 64)
    _test_torch_cat2(x, y, 1)

def test_cat_3_dif_sizes_dim2():
    x = torch.randn(256, 12, 64)
    y = torch.randn(256, 12, 64)
    _test_torch_cat2(x, y, 2)

def test_cat_3_dif_sizes_dim3():
     x = torch.randn(256, 12, 64)
     y = torch.randn(256, 12, 64)
     _test_torch_cat2(x, y, -1)

#@settings(deadline=None)
#@given(inputs=hu.tensors(n=3, min_dim=1, max_dim=3))
#def test_cat_hypothesis(inputs):
#    x1 = torch.tensor(inputs[0])
#    x2 = torch.tensor(inputs[1])
#    x3 = torch.tensor(inputs[2])
#    _test_torch_cat(x1, x2, x3, 0)
#    _test_torch_cat(x1, x2, x3, 1)
#    _test_torch_cat(x1, x2, x3, 2)

