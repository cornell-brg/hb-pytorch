""" tests for tanh and tanh.out kernels
5/26/2020 Jack Weber (jlw422@cornell.edu)
"""

import math
import torch
import torch.nn as nn
from hypothesis import given, settings

def test_tanh_1():
    a = torch.rand(10)
    out = torch.tanh(a.hammerblade())
    i = 0
    while (i<out.numel()):
        assert(out[i] == math.tanh(a[i]))
