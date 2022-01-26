"""
Unit tests for torch.max
11/14/2021 Aditi Agarwal (aa2224@cornell.edu)
01/25/2022 Zhongyuan Zhao (zz546@cornell.edu)
"""

import torch
import random
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_max(tensor, dim=None, keepdim=False):
    tensor_h = tensor.hammerblade()
    if dim is None:
        cpu_max = torch.max(tensor)
        hb_max = torch.max(tensor_h)
        assert hb_max.device == torch.device("hammerblade")
        assert torch.allclose(hb_max.cpu(), cpu_max)
    else:
        cpu_values, cpu_indices = torch.max(tensor, dim, keepdim=keepdim)
        hb_values, hb_indices = torch.max(tensor_h, dim, keepdim=keepdim)
        assert hb_values.device == torch.device("hammerblade")
        assert hb_indices.device == torch.device("hammerblade")
        assert torch.allclose(hb_values.cpu(), cpu_values)
        assert torch.allclose(hb_indices.cpu(), cpu_indices)

def test_torch_max_1():
    x = torch.ones(10)
    _test_torch_max(x)

def test_torch_max_2():
    x = torch.ones(10)
    _test_torch_max(x, dim=0)

def test_torch_max_3():
    x = torch.ones(10)
    _test_torch_max(x, dim=0, keepdim=True)

def test_torch_max_4():
    x = torch.randn(3, 4)
    _test_torch_max(x)

def test_torch_max_5():
    x = torch.randn(3, 4)
    _test_torch_max(x, dim=0)

def test_torch_max_6():
    x = torch.randn(3, 4)
    _test_torch_max(x, dim=0, keepdim=True)

def test_torch_max_7():
    x = torch.randn(3, 4)
    _test_torch_max(x, dim=1)

def test_torch_max_8():
    x = torch.randn(3, 4)
    _test_torch_max(x, dim=1, keepdim=True)

def test_torch_max_9():
    x = torch.randn(3, 4, 5)
    _test_torch_max(x)

def test_torch_max_10():
    x = torch.randn(3, 4, 5)
    _test_torch_max(x, dim=0)

def test_torch_max_11():
    x = torch.randn(3, 4, 5)
    _test_torch_max(x, dim=0, keepdim=True)

def test_torch_max_12():
    x = torch.randn(3, 4, 5)
    _test_torch_max(x, dim=1)

def test_torch_max_13():
    x = torch.randn(3, 4, 5)
    _test_torch_max(x, dim=1, keepdim=True)

def test_torch_max_14():
    x = torch.randn(3, 4, 5)
    _test_torch_max(x, dim=2)

def test_torch_max_15():
    x = torch.randn(3, 4, 5)
    _test_torch_max(x, dim=2, keepdim=True)

def test_torch_max_26():
    x = torch.tensor([[[1.], [2.], [3.]]])
    h = x.hammerblade()
    x = x.expand(2, 3, 4)
    h = h.expand(2, 3, 4)
    assert h.device == torch.device("hammerblade")
    assert not h.is_contiguous()
    max_ = torch.max(h)
    assert max_.device == torch.device("hammerblade")
    assert torch.allclose(max_.cpu(), torch.max(x))

def test_torch_max_29():
    x = torch.rand(2, 32, 64, 5)
    _test_torch_max(x)

def test_torch_max_30():
    x = torch.rand(2, 32, 64, 5)
    for dim in range(4):
        _test_torch_max(x, dim=dim)

def test_torch_max_31():
    x = torch.rand(1, 10)
    _test_torch_max(x, dim=0)

def test_torch_max_32():
    x = torch.rand(1, 3, 4)
    _test_torch_max(x, dim=0)

@settings(deadline=None)
@given(tensor=hu.tensor())
def test_torch_max_hypothesis(tensor):
    x = torch.tensor(tensor)
    _test_torch_max(x)
    for dim in range(x.dim()):
        _test_torch_max(x, dim=dim)

@pytest.mark.skipif(not torch.hb_emul_on, reason="Prohibitively slow on cosim")
def test_large_index():
    x = torch.randn(128, 850, 200)
    _test_torch_max(x, dim=1)
