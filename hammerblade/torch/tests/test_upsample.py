"""
Tests on torch.nn.Upsample and its backward
01/06/2021 Lin Cheng
Currently only 1D is supported
Currently only nearest neighbor is supported
Upsample1D takes in 3D tensors -- (N,Channel,Elem)
"""

import torch
import torch.nn as nn
import random
import hbutils
import pytest

torch.manual_seed(42)
random.seed(42)

def _test_upsample1d_nearest(x, scale_factor):
  x = x.clone().detach().requires_grad_(True)
  x_hb = hbutils.init_hb_tensor(x)
  assert x is not x_hb

  m    = nn.Upsample(scale_factor=scale_factor, mode='nearest')
  m_hb = nn.Upsample(scale_factor=scale_factor, mode='nearest')

  y    = m(x)
  y_hb = m_hb(x_hb)

  assert torch.equal(y, y_hb.cpu())

  grad    = y.clone().detach()
  grad_hb = y_hb.clone().detach()
  y.backward(grad)
  y_hb.backward(grad_hb)

  assert x.grad is not None
  assert x_hb.grad is not None
  assert x.grad is not x_hb.grad
  assert torch.allclose(x.grad, x_hb.grad.cpu())

def test_upsample1d_nearest1():
  x = torch.ones(1,1,1)
  scale_factor = 2
  _test_upsample1d_nearest(x, scale_factor)

def test_upsample1d_nearest2():
  x = torch.ones(1,1,128)
  scale_factor = 2
  _test_upsample1d_nearest(x, scale_factor)

def test_upsample1d_nearest3():
  x = torch.rand(1,1,128)
  scale_factor = 3
  _test_upsample1d_nearest(x, scale_factor)

def test_upsample1d_nearest4():
  x = torch.rand(2,3,128)
  scale_factor = 2
  _test_upsample1d_nearest(x, scale_factor)

def test_upsample1d_nearest5():
  x = torch.rand(2,3,128)
  scale_factor = 3
  _test_upsample1d_nearest(x, scale_factor)
