"""
Unit tests for torch.fill_ kernel
03/06/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch

def test_torch_fill_1():
    x = torch.empty(1,10)
    x_h = x.hammerblade()
    x.fill_(42)
    x_h.fill_(42)
    assert x_h.device == torch.device("hammerblade")
    assert torch.equal(x_h.cpu(), x)

def test_torch_fill_2():
    x = torch.empty(5,6)
    x_h = x.hammerblade()
    x.fill_(42)
    x_h.fill_(42)
    assert x_h.device == torch.device("hammerblade")
    assert torch.equal(x_h.cpu(), x)
