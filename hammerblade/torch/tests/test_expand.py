"""
Tests on torch.contiguous
03/18/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch

def test_torch_expand_1():
    x = torch.tensor([[1.], [2.], [3.]])
    x_h = x.hammerblade()
    y = x.expand(3, 4)
    y_h = x_h.expand(3, 4)
    assert y_h.device == torch.device("hammerblade")
    assert not y_h.is_contiguous()
    assert torch.equal(y_h.cpu(), y)

def test_torch_expand_2():
    x = torch.randn(3, 1)
    x_h = x.hammerblade()
    y = x.expand(3, 4)
    y_h = x_h.expand(3, 4)
    assert y_h.device == torch.device("hammerblade")
    assert not y_h.is_contiguous()
    assert torch.equal(y_h.cpu(), y)

def test_torch_expand_as_1():
    x = torch.tensor([[1.], [2.], [3.]])
    t = torch.randn(3, 4)
    x_h = x.hammerblade()
    y = x.expand_as(t)
    y_h = x_h.expand_as(t)
    assert y_h.device == torch.device("hammerblade")
    assert not y_h.is_contiguous()
    assert torch.equal(y_h.cpu(), y)

def test_torch_expand_as_2():
    x = torch.randn(3, 1)
    t = torch.randn(3, 4)
    x_h = x.hammerblade()
    y = x.expand_as(t)
    y_h = x_h.expand_as(t)
    assert y_h.device == torch.device("hammerblade")
    assert not y_h.is_contiguous()
    assert torch.equal(y_h.cpu(), y)

def test_torch_expand_as_3():
    x = torch.tensor([[1.], [2.], [3.]])
    t = torch.randn(3, 4).hammerblade()
    x_h = x.hammerblade()
    y = x.expand_as(t)
    y_h = x_h.expand_as(t)
    assert y_h.device == torch.device("hammerblade")
    assert not y_h.is_contiguous()
    assert torch.equal(y_h.cpu(), y)

def test_torch_expand_as_4():
    x = torch.randn(3, 1)
    t = torch.randn(3, 4).hammerblade()
    x_h = x.hammerblade()
    y = x.expand_as(t)
    y_h = x_h.expand_as(t)
    assert y_h.device == torch.device("hammerblade")
    assert not y_h.is_contiguous()
    assert torch.equal(y_h.cpu(), y)
