"""
tests of simple cat kernel.
"""
import torch
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

def _test_torch_cat(x, y, z):
    x_h = x.hammerblade()
    y_h = y.hammerblade()
    z_h = z.hammerblade()
    a = torch.cat([x, y, z], 0)
    a_h = torch.cat([x_h, y_h, z_h], 0)
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y, y_h.cpu())

def test_cat_1():
    x = torch.ones(10)
    _test_torch_cat(x, x, x)

def test_cat_1_dif_sizes():
    x = torch.randn(3)
    y = torch.randn(2)
    z = torch.tensor([])
    _test_torch_cat(x, y, z)

@settings(deadline=None)
@given(inputs=hu.tensors1d(n=3))
def test_cat_1_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    x3 = torch.tensor(inputs[2])
    _test_torch_cat(x1, x2, x3)

def test_cat_2():
    x = torch.randn(3, 4)
    _test_torch_cat(x, x, x)

def test_cat_2_dif_sizes():
    x = torch.randn(3, 4)
    y = torch.randn(2, 4)
    z = torch.randn(4, 4)
    _test_torch_cat(x, y, z)

@settings(deadline=None)
@given(inputs=hu.tensors(n=3, min_dim=2, max_dim=2))
def test_cat_2_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    x3 = torch.tensor(inputs[2])
    _test_torch_cat(x1, x2, x3)

def test_cat_3():
    x = torch.randn(3, 4, 5)
    _test_torch_cat(x, x, x)

def test_cat_3_dif_sizes():
    x = torch.randn(3, 4, 5)
    y = torch.randn(2, 4, 5)
    z = torch.randn(4, 4, 5)
    _test_torch_cat(x, y, z)

@settings(deadline=None)
@given(inputs=hu.tensors(n=3, min_dim=3, max_dim=3))
def test_cat_3_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    x3 = torch.tensor(inputs[2])
    _test_torch_cat(x1, x2, x3)

def test_cat_error_1():
    x = torch.randn(3, 4, 5, 2).hammerblade()
    with pytest.raises(RuntimeError):
        torch.cat([x, x, x], 0)

def test_cat_erorr_2():
    x = torch.randn(3, 4).hammerblade()
    with pytest.raises(RuntimeError):
        torch.cat([x, x, x], 1)

def test_cat_error_3():
    with pytest.raises(RuntimeError):
        torch.cat([], 0)

def test_cat_error_4():
    x = torch.ones(2).hammerblade()
    y = torch.randn(3, 4).hammerblade()
    with pytest.raises(RuntimeError):
        torch.cat([x, y], 0)
