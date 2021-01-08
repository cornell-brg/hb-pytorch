"""
Tests on torch.nn.NLLLoss2d
01/08/2021 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn as nn
import random
import hbutils

torch.manual_seed(42)
random.seed(42)


# in this case, we only support (N,C,Elm)
# we only support mean as reduction

def _test_torch_nn_NLLLoss2d(x, y):
  x = x.clone().detach().requires_grad_(True)
  x_hb = hbutils.init_hb_tensor(x)
  y = y.clone().detach().requires_grad_(False)
  y_hb = y.hammerblade()
  assert x is not x_hb
  assert y is not y_hb

  m    = nn.NLLLoss(reduction='mean')
  m_hb = nn.NLLLoss(reduction='mean')
  assert m is not m_hb

  loss    = m(x, y)
  loss_hb = m_hb(x_hb, y_hb)

  assert torch.allclose(loss, loss_hb.cpu())

  loss.backward()
  loss_hb.backward()

  assert x.grad is not None
  assert x_hb.grad is not None
  assert x.grad is not x_hb.grad
  assert torch.allclose(x.grad, x_hb.grad.cpu())

def test_torch_nn_NLLLoss2d_1():
  N, C, E = 1, 1, 10
  data = torch.ones(N,C,E)
  m = nn.LogSoftmax(dim=1)
  target = torch.zeros(N, E,dtype=torch.long)
  _test_torch_nn_NLLLoss2d(m(data), target)

def test_torch_nn_NLLLoss2d_2():
  N, C, E = 1, 2, 10
  data = torch.ones(N,C,E)
  m = nn.LogSoftmax(dim=1)
  target = torch.zeros(N, E,dtype=torch.long)
  _test_torch_nn_NLLLoss2d(m(data), target)

def test_torch_nn_NLLLoss2d_3():
  N, C, E = 1, 2, 10
  data = torch.randn(N,C,E)
  m = nn.LogSoftmax(dim=1)
  target = torch.empty(N, E,dtype=torch.long).random_(0, C)
  _test_torch_nn_NLLLoss2d(m(data), target)

def test_torch_nn_NLLLoss2d_4():
  N, C, E = 3, 4, 20
  data = torch.randn(N,C,E)
  m = nn.LogSoftmax(dim=1)
  target = torch.empty(N, E,dtype=torch.long).random_(0, C)
  _test_torch_nn_NLLLoss2d(m(data), target)
