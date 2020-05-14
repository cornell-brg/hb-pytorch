"""
Tests on torch.contiguous
03/18/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random

torch.manual_seed(42)
random.seed(42)

def _test_torch_contiguous(x):
    h = x.hammerblade()
    y = h.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x, y.cpu())

# These tensors are already contiguous
def test_torch_contiguous_1():
    x = torch.ones(10)
    _test_torch_contiguous(x)

def test_torch_contiguous_2():
    x = torch.randn(10)
    _test_torch_contiguous(x)

def test_torch_contiguous_3():
    x = torch.randn(3, 4)
    _test_torch_contiguous(x)

# transpose makes a tensor non contiguous
def test_torch_contiguous_4():
    x = torch.randn(3, 4)
    h = x.hammerblade().t()
    assert not h.is_contiguous()
    y = h.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x.t(), y.cpu())

def test_torch_contiguous_5():
    x = torch.randn(3, 3)
    h = x.hammerblade().t()
    assert not h.is_contiguous()
    y = h.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x.t(), y.cpu())

def test_torch_contiguous_6():
    x = torch.randn(2, 9)
    h = x.hammerblade().t()
    assert not h.is_contiguous()
    y = h.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x.t(), y.cpu())
