"""
Zhang-Group tests on Pytorch => tests of offloading sparse kernels
March 22, 2020
Zhongyuan Zhao
"""

import torch

def test_dense_sparse_elementwise_add_1():
    xd = torch.ones(1, 100)
    xs = torch.rand(1, 100).to_sparse()
    xr = torch.add(xd, xs)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.add(hb_xd, hb_xs)
    
    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_dense_sparse_elementwise_add_2():
    xd = torch.rand(16, 32)
    xs = torch.rand(16, 32).to_sparse()
    xr = torch.add(xd, xs)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.add(hb_xd, hb_xs)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_dense_sparse_elementwise_add_3():
    xd = torch.rand(10, 10, 16)
    xs = torch.rand(10, 10, 16).to_sparse()
    xr = torch.add(xd, xs)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.add(hb_xd, hb_xs)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_dense_sparse_elelemtwise_sub_1():
    xd = torch.ones(1, 100)
    xs = torch.rand(1, 100).to_sparse()
    xr = torch.sub(xd, xs)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.sub(hb_xd, hb_xs)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_dense_sparse_elelemtwise_sub_2():
    xd = torch.rand(16, 32)
    xs = torch.rand(16, 32).to_sparse()
    xr = torch.sub(xd, xs)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.sub(hb_xd, hb_xs)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_dense_sparse_elementwise_sub_3():
    xd = torch.ones(10, 10, 16)
    xs = torch.rand(10, 10, 16).to_sparse()
    xr = torch.sub(xd, xs)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.sub(hb_xd, hb_xs)
    print(hb_xr)
    cpu_r = hb_xr.to("cpu")

    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_customized_dense_sparse_tensor_add():

    i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
    v = torch.FloatTensor([3, 4, 5])
    x = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
    xs = x.coalesce()
    xd = torch.ones(2,3)
    xr = torch.add(xd, xs)

    hb_xs = xs.hammerblade()
    hb_xd = xd.hammerblade()
    hb_xr = torch.add(hb_xd, hb_xs)
    
    cpu_r = hb_xr.cpu()
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_customized_dense_sparse_tensor_sub():

    i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
    v = torch.FloatTensor([3, 4, 5])
    x = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
    xs = x.coalesce()
    xd = torch.ones(2,3)
    xr = torch.sub(xd, xs)

    hb_xs = xs.hammerblade()
    hb_xd = xd.hammerblade()
    hb_xr = torch.sub(hb_xd, hb_xs)

    cpu_r = hb_xr.cpu()
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)
