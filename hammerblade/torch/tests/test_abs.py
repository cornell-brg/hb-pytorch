"""
Unit tests for torch.abs kernel
03/06/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch

def _test_torch_asb(x):
    h = x.hammerblade()
    abs_x = x.abs()
    abs_h = h.abs()
    assert abs_h.device == torch.device("hammerblade")
    assert torch.equal(abs_h.cpu(), abs_x)

def test_torch_abs_1():
    x = torch.ones(10)
    _test_torch_asb(x)

def test_torch_abs_2():
    x = torch.randn(3, 4)
    _test_torch_asb(x)
