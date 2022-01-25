"""
Unit tests for torch.prod
01/25/2022 Zhongyuan Zhao (zz546@cornell.edu)
"""

import torch
import random
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_prod(tensor, dim=None, keepdim=False):
    tensor_h = tensor.hammerblade()
    if dim is None:
        prod_ = torch.prod(tensor_h)
        assert prod_.device == torch.device("hammerblade")
        assert torch.allclose(prod_.cpu(), torch.prod(tensor), atol=1e-06)
    else:
        prod_ = torch.prod(tensor_h, dim, keepdim=keepdim)
        assert prod_.device == torch.device("hammerblade")
        assert torch.allclose(prod_.cpu(), torch.prod(tensor, dim, keepdim=keepdim), atol=1e-06)

def test_torch_prod_1():
    x = torch.ones(10)
    _test_torch_prod(x)

def test_torch_prod_2():
    x = torch.ones(10)
    _test_torch_prod(x, dim=0)

def test_torch_prod_3():
    x = torch.ones(10)
    _test_torch_prod(x, dim=0, keepdim=True)

def test_torch_prod_4():
    x = torch.randn(3, 4)
    _test_torch_prod(x)

def test_torch_prod_5():
    x = torch.randn(3, 4)
    _test_torch_prod(x, dim=0)

def test_torch_prod_6():
    x = torch.randn(3, 4)
    _test_torch_prod(x, dim=0, keepdim=True)

def test_torch_prod_7():
    x = torch.randn(3, 4)
    _test_torch_prod(x, dim=1)

def test_torch_prod_8():
    x = torch.randn(3, 4)
    _test_torch_prod(x, dim=1, keepdim=True)

def test_torch_prod_11():
    x = torch.randn(3, 4, 5)
    _test_torch_prod(x)

def test_torch_prod_12():
    x = torch.randn(3, 4, 5)
    _test_torch_prod(x, dim=0)

def test_torch_prod_13():
    x = torch.randn(3, 4, 5)
    _test_torch_prod(x, dim=0, keepdim=True)

def test_torch_prod_14():
    x = torch.randn(3, 4, 5)
    _test_torch_prod(x, dim=1)

def test_torch_prod_15():
    x = torch.randn(3, 4, 5)
    _test_torch_prod(x, dim=1, keepdim=True)

def test_torch_prod_16():
    x = torch.randn(3, 4, 5)
    _test_torch_prod(x, dim=2)

def test_torch_prod_17():
    x = torch.randn(3, 4, 5)
    _test_torch_prod(x, dim=2, keepdim=True)

def test_torch_prod_26():
    x = torch.tensor([[[1.], [2.], [3.]]])
    h = x.hammerblade()
    x = x.expand(2, 3, 4)
    h = h.expand(2, 3, 4)
    assert h.device == torch.device("hammerblade")
    assert not h.is_contiguous()
    prod_ = torch.prod(h)
    assert prod_.device == torch.device("hammerblade")
    assert torch.allclose(prod_.cpu(), torch.prod(x))

def test_torch_prod_29():
    x = torch.rand(2, 32, 64, 5)
    _test_torch_prod(x)

def test_torch_prod_30():
    x = torch.rand(2, 32, 64, 5)
    for dim in range(4):
        _test_torch_prod(x, dim=dim)

def test_torch_prod_31():
    x = torch.rand(1, 10)
    _test_torch_prod(x, dim=0)

def test_torch_prod_32():
    x = torch.rand(1, 3, 4)
    _test_torch_prod(x, dim=0)

def test_torch_prod_33():
    x = torch.tensor([[1.]])
    h = x.hammerblade()
    x = x.expand(1, 10)
    h = h.expand(1, 10)
    assert h.device == torch.device("hammerblade")
    assert not h.is_contiguous()
    prod_ = torch.prod(h, 0, keepdim=True)
    assert prod_.device == torch.device("hammerblade")
    assert torch.allclose(prod_.cpu(), torch.prod(x, 0, keepdim=True))

@settings(deadline=None)
@given(tensor=hu.tensor())
def test_torch_prod_hypothesis(tensor):
    x = torch.tensor(tensor)
    _test_torch_prod(x)
    for dim in range(x.dim()):
        _test_torch_prod(x, dim=dim)

@pytest.mark.skipif(not torch.hb_emul_on, reason="Prohibitively slow on cosim")
def test_large_index():
    x = torch.randn(128, 850, 200)
    _test_torch_prod(x, dim=1)
