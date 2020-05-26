"""
Unit tests for torch.addmm kernel
03/09/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

# helper function for testing
def _test_torch_addmm(M, mat1, mat2, rel_tol=1e-05, abs_tol=1e-08):
    M_h = M.hammerblade()
    mat1_h = mat1.hammerblade()
    mat2_h = mat2.hammerblade()
    out = torch.addmm(M, mat1, mat2)
    out_h = torch.addmm(M_h, mat1_h, mat2_h)
    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(out_h.cpu(), out, rtol=rel_tol, atol=abs_tol)

def test_torch_addmm_perf():
    _test_torch_addmm(torch.randn(16, 16), torch.randn(16, 16), torch.randn(16, 16), 1e-03, 1e-06)

def test_torch_addmm_tt():
    M = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    mat1 = torch.tensor([[7., 8., 9.], [10., 11., 12.]])
    mat2 = torch.tensor([[13., 14., 15.], [16., 17., 18.], [19., 20., 21.]])
    _test_torch_addmm(M, mat1, mat2)

def test_torch_addmm_basic1():
    _test_torch_addmm(torch.ones(2, 3), torch.ones(2, 3), torch.ones(3, 3))

def test_torch_addmm_basic2():
    _test_torch_addmm(torch.ones(16, 16), torch.randn(16, 16), torch.ones(16, 16))

# 1x1 matrices
def test_torch_addmm_1x1():
    _test_torch_addmm(torch.randn(1, 1), torch.randn(1, 1), torch.randn(1, 1))

# 1x2 matrices
def test_torch_addmm_1x2():
    _test_torch_addmm(torch.randn(1, 2), torch.randn(1, 2), torch.randn(2, 2))

# broadcast self to result
def test_torch_addmm_broadcast():
    _test_torch_addmm(torch.randn(1, 3), torch.randn(2, 3), torch.randn(3, 3))

# bigger matrix
def test_torch_addmm_big_rand():
    _test_torch_addmm(torch.randn(1, 1), torch.rand(78, 56), torch.rand(56, 39), rel_tol=1e-04, abs_tol=1e-06)

# directed test with given numbers
def test_torch_addmm_directed():
    M = torch.tensor([[1., 1., 1.]])
    mat1 = torch.tensor([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.]])
    mat2 = torch.tensor([[16., 17., 18.], [19., 20., 21.], [22., 23., 24.], [25., 26., 27], [28., 29., 30.]])
    _test_torch_addmm(M, mat1, mat2)

# 1x11 matrix with just zeros and ones
def test_torch_addmm_1x11_zeros_ones():
    _test_torch_addmm(torch.zeros(1, 11), torch.ones(1, 1), torch.ones(1, 11))

# 1x11 matrix with given numbers
def test_torch_addmm_1x11_given():
    M = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]])
    mat1 = torch.tensor([[12.]])
    mat2 = torch.tensor([[13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.]])
    _test_torch_addmm(M, mat1, mat2)

# 1x11 matrix with random numbers
def test_torch_addmm_1x11_random():
    _test_torch_addmm(torch.randn(1, 11), torch.randn(1, 1), torch.randn(1, 11))

# hypothesis testing
@settings(deadline=None)
@given(inputs=hu.tensors(n=2, min_dim=2, max_dim=2))
def test_torch_addmm_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1]).T
    M = torch.randn(x1.size(0), x2.size(1))
    _test_torch_addmm(M, x1, x2)
