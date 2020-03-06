import pytest

"""
Zhang-Group tests on Pytorch => mainly used to test sparse tensor on HammerBlade device
March 1, 2020
Zhongyuan Zhao
"""

#First step, initialize en empty sparse tensor on the hammerblade DRAM.
def test_sparse_hammerbalde_empty_path1():
  import torch
  sphb = torch.device("hammerblade")
  x = torch.empty((1,5), device=sphb).to_sparse()
  assert x.device == sphb
  assert x.type() == 'torch.hammerblade.sparse.FloatTensor'
  assert x.is_hammerblade


def test_sparse_hammerblade_empty_path2():
  import torch
  sphb = torch.device("hammerblade")
  x = torch.empty((1,5), device="hammerblade").to_sparse()
  assert x.device == sphb
  assert x.type() == 'torch.hammerblade.sparse.FloatTensor'
  assert x.is_hammerblade


def test_move_sparsetensor_between_cup_and_hammerblade_path1():
  import torch
  i = torch.LongTensor([[0,1,1],[2,0,2]])
  v = torch.FloatTensor([3, 4, 5])
  hb = torch.device("hammerblade")
  hb_x = torch.sparse.FloatTensor(i, v, torch.Size([2,3])).hammerblade()
  assert hb_x.device == hb
  assert hb_x.type() == 'torch.hammerblade.sparse.FloadTensor'
  assert hb_x.is_hammerblade


def test_move_sparsetensor_between_cup_and_hammerblade_path2():
  import torch
  i = torch.LongTensor([[0,1,1],[2,0,2]])
  v = torch.FloatTensor([3, 4, 5])
  hb = torch.device("hammerblade")
  hb_x = torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to("hammerblade")
  assert hb_x.device == hb
  assert hb_x.type() == 'torch.hammerblade.sparse.FloadTensor'
  assert hb_x.is_hammerblade
