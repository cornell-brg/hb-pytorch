"""
Unit tests for maxpool2d operator
03/18/2020 Bandhav Veluri
"""

import torch
import torch.nn.functional as F
import os
import pytest

def _test_max_pool2d(x, kernel_size, stride=None, padding=1):
    x_hb = x.hammerblade()

    y, r = F.max_pool2d(x, kernel_size, stride, padding, return_indices=True)
    y_hb, r_hb =  F.max_pool2d(x_hb, kernel_size, stride, padding,
                               return_indices=True)

    assert torch.allclose(y, y_hb.cpu())
    assert torch.equal(r, r_hb.cpu())

def test_max_pool2d_1():
    x = torch.rand(1, 1, 5, 5)
    kernel_size = 3
    stride = 2
    padding = 0
    _test_max_pool2d(x, kernel_size, stride, padding)

def test_max_pool2d_2():
    x = torch.ones(1, 1, 5, 5)
    kernel_size = 3
    stride = 2
    padding = 0
    _test_max_pool2d(x, kernel_size, stride, padding)

if __name__ == "__main__":
    test_max_pool2d_1()
