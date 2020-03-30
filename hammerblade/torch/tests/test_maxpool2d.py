"""
Unit tests for maxpool2d operator
03/18/2020 Bandhav Veluri
"""

import torch
import torch.nn.functional as F
import hbutils

def _test_max_pool2d(x, kernel_size, stride=None, padding=1):
    x_hb = hbutils.init_hb_tensor(x)

    y, r = F.max_pool2d(x, kernel_size, stride, padding, return_indices=True)
    y_hb, r_hb = F.max_pool2d(x_hb, kernel_size, stride, padding,
                              return_indices=True)

    assert torch.allclose(y, y_hb.cpu())
    assert torch.equal(r.type(torch.int), r_hb.cpu())

    if x.requires_grad:
        grad = torch.rand(y.shape)
        grad_hb = grad.hammerblade()

        y.backward(grad)
        y_hb.backward(grad_hb)

        assert torch.equal(x.grad, x_hb.grad.cpu())

def test_max_pool2d_1():
    x = torch.rand(1, 1, 5, 5, requires_grad=True)
    kernel_size = 3
    stride = 2
    padding = 0
    _test_max_pool2d(x, kernel_size, stride, padding)

def test_max_pool2d_2():
    """
    All elements same
    """
    x = torch.ones(1, 1, 5, 5, requires_grad=True)
    kernel_size = 3
    stride = 2
    padding = 0
    _test_max_pool2d(x, kernel_size, stride, padding)

def test_max_pool2d_3():
    """
    Multi channel
    """
    x = torch.rand(1, 3, 5, 5, requires_grad=True)
    kernel_size = 3
    stride = 2
    padding = 0
    _test_max_pool2d(x, kernel_size, stride, padding)

def test_max_pool2d_4():
    """
    Multi batch multi channel
    """
    x = torch.rand(2, 3, 5, 5, requires_grad=True)
    kernel_size = 3
    stride = 2
    padding = 0
    _test_max_pool2d(x, kernel_size, stride, padding)

def test_max_pool2d_5():
    """
    Multi batch multi channel padding
    """
    x = torch.rand(2, 3, 5, 5, requires_grad=True)
    kernel_size = 2
    stride = 2
    padding = 1
    _test_max_pool2d(x, kernel_size, stride, padding)

def test_max_pool2d_6():
    """
    Multi batch multi channel assymentric kernel
    """
    x = torch.rand(2, 2, 10, 10, requires_grad=True)
    kernel_size = (6, 7)
    stride = 2
    padding = 2
    _test_max_pool2d(x, kernel_size, stride, padding)

def test_max_pool2d_7():
    """
    Multi batch multi channel assymentric
    """
    x = torch.rand(2, 2, 10, 10, requires_grad=True)
    kernel_size = (7, 6)
    stride = (1, 2)
    padding = (2, 3)
    _test_max_pool2d(x, kernel_size, stride, padding)

if __name__ == "__main__":
    test_max_pool2d_1()
