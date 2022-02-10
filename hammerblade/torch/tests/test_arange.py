"""
Unit test on reduction arange kernel
01/25/2022 Zhongyuan Zhao (zz546@cornell.edu)
"""

import torch
import random
from math import isnan, isinf
from hypothesis import assume, given, settings
import hypothesis.strategies as st
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

# test of arange two tensors

def _test_arange(x, start, end, step):
    hb_x = x.hammerblade()
    assert hb_x is not x
    x = torch.arange(x, start, end, step, dtype=int)
    hb_x = torch.arange(hb_x, start, end, step, dtype=int)
    assert hb_x.device == torch.device("hammerblade")
    assert torch.allclose(x, hb_x.cpu())

def test_arange_int_1():
    x = torch.zeros(64).int()
    _test_arange(x, 0, 64, 1)

def test_arange_int_2():
    x = torch.zeros(64).int()
    _test_arange(x, 0, 32, 1)

def test_arange_int_3():
    x = torch.zeros(64).int()
    _test_arange(x, 0, 64, 2)

def test_arange_int_4():
    x = torch.zeros(64).int()
    _test_arange(x, 0, 32, 2)

def test_arange_long_1():
    x = torch.zeros(64)
    _test_arange(x, 0, 64, 1)

def test_arange_long_2():
    x = torch.zeros(64)
    _test_arange(x, 0, 32, 1)

def test_arange_long_3():
    x = torch.zeros(64)
    _test_arange(x, 0, 64, 2)

def test_arange_long_4():
    x = torch.zeros(64)
    _test_arange(x, 0, 32, 2)


