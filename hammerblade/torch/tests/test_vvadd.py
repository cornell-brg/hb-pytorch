"""
[TUTORIAL]
Unit tests for torch.vvadd kernel
06/04/2020 Krithik Ranjan (kr397@cornell.edu)
"""

import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_vvadd(t1, t2):
    h1 = t1.hammerblade()
    h2 = t2.hammerblade()
    out = t1 + t2
    out_h = torch.vvadd(h1, h2)
    assert out_h.device == torch.device('hammerblade')
    assert torch.allclose(out_h.cpu(), out)

def test_torch_vvadd_1():
    t1 = torch.ones(10)
    t2 = torch.ones(10)
    _test_torch_vvadd(t1, t2)

def test_torch_vvadd_2():
    t1 = torch.randn(10)
    t2 = torch.randn(10)
    _test_torch_vvadd(t1, t2)

@settings(deadline=None)
@given(tensors=hu.tensors(n=2))
def test_torch_addcmul_hypothesis(tensors):
    t1 = torch.tensor(tensors[0])
    t2 = torch.tensor(tensors[1])
    _test_torch_vvadd(t1, t2)
