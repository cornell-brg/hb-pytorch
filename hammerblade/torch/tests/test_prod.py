"""
Unit tests for torch.prod
11/14/2021 Aditi Agarwal (aa2224@cornell.edu)
"""

import torch
import random
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)


# ------------------------------------------------------------------------
# test of getting product of tensor t along dimension dim
# ------------------------------------------------------------------------

def _test_torch_prod(t,dim):
    h = t.hammerblade()
    assert h is not t
    out = torch.prod(t,dim)
    out_h = torch.prod(h,dim)
    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(out_h.cpu(), out)   


def test_rand_tensor_x():
    t = torch.randn(10)
    _test_torch_prod(t,0)

def test_rand_tensor_y():
    t = torch.randn(10)
    _test_torch_prod(t,1)