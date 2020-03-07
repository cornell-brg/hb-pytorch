"""
Unit tests for torch.abs kernel
03/06/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch

def test_torch_abs_1():
  x = torch.randn(10)
  x_h = x.hammerblade()
  abs_x = x.abs()
  abs_x_h = x_h.abs()
  assert abs_x_h.device == torch.device("hammerblade")
  assert torch.equal(abs_x_h.cpu(), abs_x)

def test_torch_abs_2():
  x = torch.randn(3, 4)
  x_h = x.hammerblade()
  abs_x = x.abs()
  abs_x_h = x_h.abs()
  assert abs_x_h.device == torch.device("hammerblade")
  assert torch.equal(abs_x_h.cpu(), abs_x)
