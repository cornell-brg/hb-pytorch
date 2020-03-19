"""
Zhang-Group tests on Pytorch => mainly used to test sparse tensor on HammerBlade device
March 1, 2020
Zhongyuan Zhao
"""

# First step, initialize en empty sparse tensor on the hammerblade DRAM.
# def test_sparse_hammerbalde_empty_path1():
#   import torch
#   sphb = torch.device("hammerblade")
#   x = torch.empty((1,128), device=sphb).to_sparse()
#   assert x.device == sphb
#   assert x.type() == 'torch.hammerblade.sparse.FloatTensor'
#   assert x.is_hammerblade


# def test_sparse_hammerblade_empty_path2():
#   import torch
#   sphb = torch.device("hammerblade")
#   x = torch.empty((1,128), device="hammerblade").to_sparse()
#   assert x.device == sphb
#   assert x.type() == 'torch.hammerblade.sparse.FloatTensor'
#   assert x.is_hammerblade


def test_sparse_hammerblade_empty_path1():
    import torch

    cpu = torch.device("cpu")
    cpu_x = torch.empty((1, 32)).to_sparse()
    cpu_i = cpu_x._indices()
    cpu_v = cpu_x._values()

    hb = torch.device("hammerblade")
    hb_x = cpu_x.to("hammerblade")
    hb_i = hb_x._indices()
    hb_v = hb_x._values()

    assert cpu_x.device == cpu
    assert cpu_x.type() == 'torch.sparse.FloatTensor'
    assert hb_x.device == hb
    assert hb_x.type() == 'torch.hammerblade.sparse.FloatTensor'
    assert hb_x.is_hammerblade
    assert hb_i.type() == 'torch.hammerblade.LongTensor'
    assert hb_v.type() == 'torch.hammerblade.FloatTensor'
    assert torch.equal(hb_i.cpu(), cpu_i)
    assert torch.equal(hb_v.cpu(), cpu_v)

def test_sparse_hammerblade_empty_path2():
    import torch

    cpu = torch.device("cpu")
    cpu_x = torch.empty((4, 10)).to_sparse()
    cpu_i = cpu_x._indices()
    cpu_v = cpu_x._values()

    hb = torch.device("hammerblade")
    hb_x = cpu_x.to("hammerblade")
    hb_i = hb_x._indices()
    hb_v = hb_x._values()

    assert cpu_x.device == cpu
    assert cpu_x.type() == 'torch.sparse.FloatTensor'
    assert hb_x.device == hb
    assert hb_x.type() == 'torch.hammerblade.sparse.FloatTensor'
    assert hb_x.is_hammerblade
    assert hb_i.type() == 'torch.hammerblade.LongTensor'
    assert hb_v.type() == 'torch.hammerblade.FloatTensor'
    assert torch.equal(hb_i.cpu(), cpu_i)
    assert torch.equal(hb_v.cpu(), cpu_v)

def test_sparse_hammerblade_rand_path():
    import torch

    cpu = torch.device("cpu")
    cpu_x = torch.rand((4, 10)).to_sparse()
    cpu_i = cpu_x._indices()
    cpu_v = cpu_x._values()

    hb = torch.device("hammerblade")
    hb_x = cpu_x.to("hammerblade")
    hb_i = hb_x._indices()
    hb_v = hb_x._values()

    assert cpu_x.device == cpu
    assert cpu_x.type() == 'torch.sparse.FloatTensor'
    assert hb_x.device == hb
    assert hb_x.type() == 'torch.hammerblade.sparse.FloatTensor'
    assert hb_x.is_hammerblade
    assert hb_i.type() == 'torch.hammerblade.LongTensor'
    assert hb_v.type() == 'torch.hammerblade.FloatTensor'
    assert torch.equal(hb_i.cpu(), cpu_i)
    assert torch.equal(hb_v.cpu(), cpu_v)

#This function tests the methods inside sparse tensor which do not need to be offloaded to a well prepared hammerblade kernel, but still need to extended these functions to SparseHammerBlade.
def test_device_guard_false_functions():
    import torch

    cpu = torch.device("cpu")
    cpu_x = torch.rand((4, 10)).to_sparse()
    cpu_i = cpu_x._indices()
    cpu_v = cpu_x._values()

    hb = torch.device("hammerblade")
    hb_x = cpu_x.to("hammerblade")
    hb_i = hb_x._indices()
    hb_v = hb_x._values()

    assert cpu_x.device == cpu
    assert cpu_x.type() == 'torch.sparse.FloatTensor'
    assert hb_x.device == hb
    assert hb_x.type() == 'torch.hammerblade.sparse.FloatTensor'
    assert hb_x.is_hammerblade
    assert hb_i.type() == 'torch.hammerblade.LongTensor'
    assert hb_v.type() == 'torch.hammerblade.FloatTensor'
    assert hb_x.is_coalesced() == cpu_x.is_coalesced()
    assert hb_x.dense_dim() == cpu_x.dense_dim()
    assert hb_x.sparse_dim() == cpu_x.sparse_dim()
    assert hb_x._nnz() == cpu_x._nnz()
    assert hb_x._dimI() == cpu_x._dimI()
    assert hb_x._dimV() == cpu_x._dimV()
    assert torch.equal(hb_i.cpu(), cpu_i)
    assert torch.equal(hb_v.cpu(), cpu_v)

def test_move_sparsetensor_between_cup_and_hammerblade_path1():
    import torch

    cpu = torch.device("cpu")
    i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
    v = torch.FloatTensor([3, 4, 5])
    cpu_x = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
    cpu_i = cpu_x._indices()
    cpu_v = cpu_x._values()

    hb = torch.device("hammerblade")
    hb_x = cpu_x.hammerblade()
    hb_i = hb_x._indices()
    hb_v = hb_x._values()

    assert cpu_x.device == cpu
    assert cpu_x.type() == 'torch.sparse.FloatTensor'
    assert hb_x.device == hb
    assert hb_x.type() == 'torch.hammerblade.sparse.FloatTensor'
    assert hb_x.is_hammerblade
    assert hb_i.type() == 'torch.hammerblade.LongTensor'
    assert hb_v.type() == 'torch.hammerblade.FloatTensor'
    assert torch.equal(hb_i.cpu(), cpu_i)
    assert torch.equal(hb_v.cpu(), cpu_v)


def test_move_sparsetensor_between_cup_and_hammerblade_path2():
    import torch

    cpu = torch.device("cpu")
    i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
    v = torch.FloatTensor([3, 4, 5])
    cpu_x = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
    cpu_i = cpu_x._indices()
    cpu_v = cpu_x._values()

    hb = torch.device("hammerblade")
    hb_x = cpu_x.to("hammerblade")
    hb_i = hb_x._indices()
    hb_v = hb_x._values()

    assert cpu_x.device == cpu
    assert cpu_x.type() == 'torch.sparse.FloatTensor'
    assert hb_x.device == hb
    assert hb_x.type() == 'torch.hammerblade.sparse.FloatTensor'
    assert hb_x.is_hammerblade
    assert hb_i.type() == 'torch.hammerblade.LongTensor'
    assert hb_v.type() == 'torch.hammerblade.FloatTensor'
    assert torch.equal(hb_i.cpu(), cpu_i)
    assert torch.equal(hb_v.cpu(), cpu_v)
