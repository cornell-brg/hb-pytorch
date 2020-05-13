"""
Unit tests for torch.addcmul kernel
04/23/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_addcmul(t1, t2, t3):
    h1 = t1.hammerblade()
    h2 = t2.hammerblade()
    h3 = t3.hammerblade()
    out = torch.addcmul(t1, t2, t3, value=0.1)
    out_h = torch.addcmul(h1, h2, h3, value=0.1)
    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(out_h.cpu(), out)

def test_torch_addcmul_1():
    t1 = torch.ones(10)
    t2 = torch.ones(10)
    t3 = torch.ones(10)
    _test_torch_addcmul(t1, t2, t3)

def test_torch_addcmul_2():
    t1 = torch.randn(2, 3)
    t2 = torch.randn(2, 3)
    t3 = torch.randn(2, 3)
    _test_torch_addcmul(t1, t2, t3)

@settings(deadline=None)
@given(tensors=hu.tensors(n=3))
def test_torch_addcmul_hypothesis(tensors):
    t1 = torch.tensor(tensors[0])
    t2 = torch.tensor(tensors[1])
    t3 = torch.tensor(tensors[2])
    _test_torch_addcmul(t1, t2, t3)
