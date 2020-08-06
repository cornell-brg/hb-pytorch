"""
tests for tanh and tanh.out kernels
Author : Jack Weber
Date   : 05/26/2020
"""


import math
import torch
import torch.nn as nn
from hypothesis import given, settings

def _test_tanh(x):
	h = x.hammerblade()
	assert h is not x
	out_cpu = torch.tanh(x)
	out = torch.tanh(h)
	assert out.is_hammerblade
    assert torch.allclose(out.cpu(), out_cpu)

def test_tanh_1():
    a = torch.randn(10)
    _test_tanh(a)

def test_tanh_2():
    a = torch.rand(1, 128)
    _test_tanh(a)

def test_tanh_3():
    a = torch.rand(16, 32)
    _test_tanh(a)
