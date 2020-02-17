import pytest

"""
BRG tests on PyTorch => tests of real offloading kernels
Feb 09, 2020
Lin Cheng
"""
import torch

# test of adding two tensors

def test_elementwise_add_1():
  x1 = torch.ones(1,10)
  x2 = torch.ones(1,10)
  y = x1 + x2
  x1_h = x1.hammerblade()
  x2_h = x2.hammerblade()
  y_h = x1_h + x2_h
  y_c = y_h.cpu()
  assert y_h.device == torch.device("hammerblade")
  assert torch.equal(y_c, y)

def test_elementwise_add_2():
  x1 = torch.ones(4,5)
  x2 = torch.ones(4,5)
  y = x1 + x2
  x1_h = x1.hammerblade()
  x2_h = x2.hammerblade()
  y_h = x1_h + x2_h
  y_c = y_h.cpu()
  assert y_h.device == torch.device("hammerblade")
  assert torch.equal(y_c, y)

def test_elementwise_add_3():
  x1 = torch.rand(1,128)
  x2 = torch.rand(1,128)
  y = x1 + x2
  x1_h = x1.hammerblade()
  x2_h = x2.hammerblade()
  y_h = x1_h + x2_h
  y_c = y_h.cpu()
  assert y_h.device == torch.device("hammerblade")
  assert torch.equal(y_c, y)

def test_elementwise_add_4():
  x1 = torch.rand(16,32)
  x2 = torch.rand(16,32)
  y = x1 + x2
  x1_h = x1.hammerblade()
  x2_h = x2.hammerblade()
  y_h = x1_h + x2_h
  y_c = y_h.cpu()
  assert y_h.device == torch.device("hammerblade")
  assert torch.equal(y_c, y)

def test_elementwise_sub_1():
  x = torch.ones(1,10)
  y = torch.ones(1,10)
  z = x - y
  z_h = x.hammerblade() - y.hammerblade()
  assert z_h.device == torch.device("hammerblade")
  assert torch.equal(z_h.cpu(), z)

def test_elementwise_sub_2():
  x = torch.ones(4,5)
  y = torch.ones(4,5)
  z = x - y
  z_h = x.hammerblade() - y.hammerblade()
  assert z_h.device == torch.device("hammerblade")
  assert torch.equal(z_h.cpu(), z)

def test_elementwise_sub_3():
  x = torch.rand(1,128)
  y = torch.rand(1,128)
  z = x - y
  z_h = x.hammerblade() - y.hammerblade()
  assert z_h.device == torch.device("hammerblade")
  assert torch.equal(z_h.cpu(), z)

def test_elementwise_sub_4():
  x = torch.rand(16,32)
  y = torch.rand(16,32)
  z = x - y
  z_h = x.hammerblade() - y.hammerblade()
  assert z_h.device == torch.device("hammerblade")
  assert torch.equal(z_h.cpu(), z)

