""" tests for tanh and tanh.out kernels
5/26/2020 Jack Weber (jlw422@cornell.edu)
"""

import math
import torch
import torch.nn as nn
from hypothesis import given, settings

def test_tanh_1():
    a = torch.randn(10)
    out_cpu = torch.tanh(a)
    out = torch.tanh(a.hammerblade())
    assert out.is_hammerblade
    assert torch.allclose(out.cpu(), out_cpu)
