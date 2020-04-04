"""
Zhang-Group tests on Pytorch => tests of offloading sparse kernels
March 22, 2020
Zhongyuan Zhao
"""

import torch

def test_sparse_elementwise_add_1():
    xd = torch.ones(1, 10)
    xs = torch.rand(1, 10).to_sparse()
    xr = torch.add(xd, xs)
    xs_v = xs._values()
    print(xd)
    print(xs_v)
    print(xr)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    print(hb_xd)
    hb_v = hb_xs._values()
    hb_i = hb_xs._indices()
    print(hb_v)
    print(hb_i)
    hb_xr = torch.add(hb_xd, hb_xs)
    print(hb_xr)
    
    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xd)
    assert torch.allclose(cpu_r, xr)

def test_sparse_elementwise_add_2():
    xd = torch.rand(4, 5)
    xs = torch.rand(4, 5).to_sparse()
    xr = torch.add(xd, xs)
    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.add(hb_xd, hb_xs)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_sparse_elelemtwise_sub_1():
    xd = torch.rand(1, 10)
    xs = torch.rand(1, 10).to_sparse()
    xr = torch.add(xd, xs)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.sub(hb_xd, hb_xs)
    
    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_sparse_elelemtwise_sub_2():
    xd = torch.rand(4, 5)
    xs = torch.rand(4, 5).to_sparse()
    xr = torch.add(xd, xs)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.add(hb_xd, hb_xs)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)
