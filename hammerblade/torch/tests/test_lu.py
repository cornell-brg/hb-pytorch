"""
Unit tests for torch.lu kernel
10/07/2020 Kexin Zheng (kz73@cornell.edu)
"""

import torch
import random
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_lu(x):
    print("input")
    print(x)
    h = x.hammerblade()
    fac_h, piv_h, infos_h = h.lu(pivot=True)#, get_infos=False)
    print("\nhammerblade factorization")
    print(fac_h)
    print("ham pivots")
    print(piv_h)
    print("ham infos")
    print(info_h)
    fac_c, piv_c, infos_c = x.lu(pivot=True, get_infos=True)
    print("cpu factorization")
    print(fac_c)
    print("cpu pivots")
    print(piv_c)
    print("cpu infos")
    print(infos_c)
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(fac_h.cpu(), fac_c)

def test_torch_lu_basic1():
    x = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    _test_torch_lu(x)


def test_torch_lu_basic2():
    x = torch.tensor([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
    _test_torch_lu(x)

'''
def test_torch_lu_2():
    x1 = torch.randn(10)
    x2 = torch.randn(10)
    _test_torch_lu(x1, x2)

def test_torch_lu_non_contiguous():
    x = torch.tensor([1.])
    x_h = x.hammerblade()
    x = x.expand(10)
    x_h = x_h.expand(10)
    x = x.lu(x)
    x_h = x_h.lu(x_h)
    assert x_h.device == torch.device("hammerblade")
    assert torch.allclose(x_h.cpu(), x)

@pytest.mark.xfail
def test_torch_lu_different_device_F():
    x = torch.ones(10)
    x_h = x.hammerblade()
    x_h = x_h.lu(x)

@pytest.mark.xfail
def test_torch_lu_mismatching_shape_F():
    x = torch.ones(10).hammerblade()
    y = torch.ones(5).hammerblade()
    torch.lu(x, y)

@settings(deadline=None)
@given(inputs=hu.tensors1d(n=2))
def test_torch_lu_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    _test_torch_lu(x1, x2)
'''
