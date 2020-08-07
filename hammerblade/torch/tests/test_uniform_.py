"""
unit tests for torch.uniform_ kernel
Author : Jack Weber
Date   : 04/14/2020
"""

import torch
import random

torch.manual_seed(42)
random.seed(42)

# ----------------------------------------------------------------------
# tests of torch.uniform_; makes sure numbers under 1
# ----------------------------------------------------------------------

def _test_torch_uniform(x):
    out = x.cpu()
    i = 0
    same = True
    assert (x.device == torch.device("hammerblade"))
    while(i < x.numel()):
        if same == True and out[i] != out[0]:
            same = False
        assert (0 < out[i] < 1)
        i += 1
    assert(same == False)

def test_torch_uniform1():
    x = torch.rand(10, device="hammerblade")
    _test_torch_uniform(x)

def test_torch_uniform2():
    x = torch.rand(100, device="hammerblade")
    _test_torch_uniform(x)
