"""
Tests on torch.contiguous
03/18/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch

# These tensors are already contiguous
def test_torch_contiguous_1():
    x = torch.ones(10).hammerblade()
    y = x.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x.cpu(), y.cpu())

def test_torch_contiguous_2():
    x = torch.randn(10).hammerblade()
    y = x.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x.cpu(), y.cpu())

def test_torch_contiguous_3():
    x = torch.randn(3, 4).hammerblade()
    y = x.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x.cpu(), y.cpu())

# transpose makes a tensor non contiguous
def test_torch_contiguous_4():
    x = torch.randn(3, 4).hammerblade().t()
    assert not x.is_contiguous()
    y = x.contiguous()
    assert y.device == torch.device("hammerblade")
    assert y.is_contiguous()
    assert torch.equal(x.cpu(), y.cpu())
