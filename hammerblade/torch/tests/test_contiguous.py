"""
Tests on torch.contiguous
03/18/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch

# These tensors are already contiguous
def test_torch_contiguous_1():
    x = torch.ones(10)
    h = x.hammerblade()
    y = h.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x, y.cpu())

def test_torch_contiguous_2():
    x = torch.randn(10)
    h = x.hammerblade()
    y = h.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x, y.cpu())

def test_torch_contiguous_3():
    x = torch.randn(3, 4)
    h = x.hammerblade()
    y = h.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x, y.cpu())

# transpose makes a tensor non contiguous
def test_torch_contiguous_4():
    x = torch.randn(3, 4)
    h = x.hammerblade().t()
    assert not h.is_contiguous()
    y = h.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x.t(), y.cpu())
