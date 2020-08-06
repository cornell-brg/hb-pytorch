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

def test_torch_uniform1():
    x = torch.rand(10, device="hammerblade")
    out = x.cpu()
    i = 0
    assert (x.device == torch.device("hammerblade"))
    while(i < x.numel()):
        assert (out[i]<1)
        i += 1;

def test_torch_uniform2():
    x = torch.rand(100, device="hammerblade")
    out = x.cpu()
    i = 0
    assert (x.device == torch.device("hammerblade"))
    while(i < x.numel()):
        assert (out[i]<1)
        i += 1;

