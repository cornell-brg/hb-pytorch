"""
Unit tests for torch.sum
03/29/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_sum(tensor, dim=None, keepdim=False):
    tensor_h = tensor.hammerblade()
    if dim is None:
        sum_ = torch.sum(tensor_h)
        assert sum_.device == torch.device("hammerblade")
        assert torch.allclose(sum_.cpu(), torch.sum(tensor), atol=1e-06)
    else:
        sum_ = torch.sum(tensor_h, dim, keepdim=keepdim)
        assert sum_.device == torch.device("hammerblade")
        assert torch.allclose(sum_.cpu(), torch.sum(tensor, dim, keepdim=keepdim), atol=1e-06)

def test_torch_sum_1():
    x = torch.ones(10)
    _test_torch_sum(x)

def test_torch_sum_2():
    x = torch.ones(10)
    _test_torch_sum(x, dim=0)

def test_torch_sum_3():
    x = torch.ones(10)
    _test_torch_sum(x, dim=0, keepdim=True)

def test_torch_sum_4():
    x = torch.randn(3, 4)
    _test_torch_sum(x)

def test_torch_sum_5():
    x = torch.randn(3, 4)
    _test_torch_sum(x, dim=0)

def test_torch_sum_6():
    x = torch.randn(3, 4)
    _test_torch_sum(x, dim=0, keepdim=True)

def test_torch_sum_7():
    x = torch.randn(3, 4)
    _test_torch_sum(x, dim=1)

def test_torch_sum_8():
    x = torch.randn(3, 4)
    _test_torch_sum(x, dim=1, keepdim=True)

def test_torch_sum_9():
    x = torch.randn(3, 4)
    _test_torch_sum(x, dim=(0, 1))

def test_torch_sum_10():
    x = torch.randn(3, 4)
    _test_torch_sum(x, dim=(0, 1), keepdim=True)

def test_torch_sum_11():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x)

def test_torch_sum_12():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=0)

def test_torch_sum_13():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=0, keepdim=True)

def test_torch_sum_14():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=1)

def test_torch_sum_15():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=1, keepdim=True)

def test_torch_sum_16():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=2)

def test_torch_sum_17():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=2, keepdim=True)

def test_torch_sum_18():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=(0, 1))

def test_torch_sum_19():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=(0, 1), keepdim=True)

def test_torch_sum_20():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=(1, 2))

def test_torch_sum_21():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=(1, 2), keepdim=True)

def test_torch_sum_22():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=(0, 2))

def test_torch_sum_23():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=(0, 2), keepdim=True)

def test_torch_sum_24():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=(0, 1, 2))

def test_torch_sum_25():
    x = torch.randn(3, 4, 5)
    _test_torch_sum(x, dim=(0, 1, 2), keepdim=True)

def test_torch_sum_26():
    x = torch.tensor([[[1.], [2.], [3.]]])
    h = x.hammerblade()
    x = x.expand(2, 3, 4)
    h = h.expand(2, 3, 4)
    assert h.device == torch.device("hammerblade")
    assert not h.is_contiguous()
    sum_ = torch.sum(h)
    assert sum_.device == torch.device("hammerblade")
    assert torch.allclose(sum_.cpu(), torch.sum(x))

def test_torch_sum_27():
    x = torch.tensor([[[1.], [2.], [3.]]])
    h = x.hammerblade()
    x = x.expand(2, 3, 4)
    h = h.expand(2, 3, 4)
    assert h.device == torch.device("hammerblade")
    assert not h.is_contiguous()
    sum_ = torch.sum(h, (0, 2))
    assert sum_.device == torch.device("hammerblade")
    assert torch.allclose(sum_.cpu(), torch.sum(x, (0, 2)))

def test_torch_sum_28():
    x = torch.tensor([[[1.], [2.], [3.]]])
    h = x.hammerblade()
    x = x.expand(2, 3, 4)
    h = h.expand(2, 3, 4)
    assert h.device == torch.device("hammerblade")
    assert not h.is_contiguous()
    sum_ = torch.sum(h, (0, 2), keepdim=True)
    assert sum_.device == torch.device("hammerblade")
    assert torch.allclose(sum_.cpu(), torch.sum(x, (0, 2), keepdim=True))

def test_torch_sum_29():
    x = torch.rand(2, 3, 4, 5)
    _test_torch_sum(x)

def test_torch_sum_30():
    x = torch.rand(2, 3, 4, 5)
    for dim in range(4):
        _test_torch_sum(x, dim=dim)

def test_torch_sum_31():
    x = torch.rand(1, 10)
    _test_torch_sum(x, dim=0)

def test_torch_sum_32():
    x = torch.rand(1, 3, 4)
    _test_torch_sum(x, dim=0)

def test_torch_sum_33():
    x = torch.tensor([[1.]])
    h = x.hammerblade()
    x = x.expand(1, 10)
    h = h.expand(1, 10)
    assert h.device == torch.device("hammerblade")
    assert not h.is_contiguous()
    sum_ = torch.sum(h, 0, keepdim=True)
    assert sum_.device == torch.device("hammerblade")
    assert torch.allclose(sum_.cpu(), torch.sum(x, 0, keepdim=True))

@settings(deadline=None)
@given(tensor=hu.tensor())
def test_torch_sum_hypothesis(tensor):
    x = torch.tensor(tensor)
    _test_torch_sum(x)
    for dim in range(x.dim()):
        _test_torch_sum(x, dim=dim)

@settings(deadline=None)
@given(tensor=hu.tensor(min_dim=3, max_dim=3))
def test_torch_sum_hypothesis_3d(tensor):
    x = torch.tensor(tensor)
    _test_torch_sum(x, dim=(0, 1))
    _test_torch_sum(x, dim=(1, 2))
    _test_torch_sum(x, dim=(0, 2))
