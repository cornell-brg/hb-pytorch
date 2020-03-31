"""
Tests on torch.nn.Dropout
03/16/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn as nn
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

def _test_torch_nn_dropout(x, p):
    dropout = nn.Dropout(p)
    h = x.hammerblade()
    d = dropout(h)
    assert d.device == torch.device("hammerblade")
    d = d.cpu().flatten()
    x = x.flatten()
    i = 0
    while(i < x.numel()):
        assert (torch.allclose(d[i], x[i] * (1.0 / (1 - p))) or d[i] == 0)
        i += 1
    assert not torch.allclose(d, x)



def test_torch_nn_dropout_1():
    x = torch.ones(10)
    _test_torch_nn_dropout(x, 0.5)

def test_torch_nn_dropout_2():
    x = torch.randn(10)
    _test_torch_nn_dropout(x, 0.5)

def test_torch_nn_dropout_3():
    x = torch.randn(10)
    _test_torch_nn_dropout(x, 0.25)

def test_torch_nn_dropout_4():
    x = torch.randn(2, 3)
    _test_torch_nn_dropout(x, 0.25)

def test_torch_nn_dropout_5():
    x = torch.randn(3, 4, 5)
    _test_torch_nn_dropout(x, 0.25)

@settings(deadline=None)
@given(tensor=hu.tensor1d(nonzero=True, min_len=5))
def test_torch_nn_dropout_hypothesis_1d(tensor):
    x = torch.tensor(tensor)
    _test_torch_nn_dropout(x, 0.75)
