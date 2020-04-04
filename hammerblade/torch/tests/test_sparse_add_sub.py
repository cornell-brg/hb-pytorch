"""
Zhang-Group tests on Pytorch => tests of offloading sparse kernels
March 22, 2020
Zhongyuan Zhao
"""

import torch

def test_sparse_elementwise_add_1():
    xd = torch.rand(1, 10)
    xs = torch.rand(1, 10).to_sparse()
    xr = torch.add(xd, xs)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.add(hb_xd, hb_xs)
    
    cpu_r = hb_xr.to("cpu")
    assert y_h.device == torch.device("hammerblade")
    assert equal(cpu_r, xr)

def test_sparse_elementwise_add_2():
    xd = torch.rand(4, 5)
    xs = torch.rand(4, 5).to_sparse()
    xr = torch.add(xd, xs)
    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.add(hb_xd, hb_xs)

    cpu_r = hb_xr.to("cpu")
    assert y_h.device == torch.device("hammerblade")
    assert equal(cpu_r, xr)

def test_sparse_elelemtwise_sub_1():
    xd = torch.rand(1, 10)
    xs = torch.rand(1, 10).to_sparse()
    xr = torch.add(xd, xs)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.sub(hb_xd, hb_xs)
    
    cpu_r = hb_xr.to("cpu")
    assert y_h.device == torch.device("hammerblade")
    assert equal(cpu_r, xr)

def test_sparse_elelemtwise_sub_1():
    xd = torch.rand(4, 5)
    xs = torch.rand(4, 5).to_sparse()
    xr = torch.add(xd, xs)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.add(hb_xd, hb_xs)

    cpu_r = hb_xr.to("cpu")
    assert y_h.device == torch.device("hammerblade")
    assert equal(cpu_r, xr)
