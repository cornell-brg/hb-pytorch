"""
Unit tests for torch.cholesky kernel
10/07/2020 Kexin Zheng (kz73@cornell.edu)
"""

import torch
import random
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

MAXSIZE = 100

torch.manual_seed(42)
random.seed(42)

def _test_torch_cholesky(A, atol=1e-8):
    h = A.hammerblade()
    L_h  = h.cholesky_hammerblade()

    # reconstruct L and L^T
    L = L_h.cpu()
    LT = L.t()

    # calculate L * L^T
    LLT = torch.mm(L, LT)

    assert torch.allclose(A, LLT, atol=atol)

def test_torch_cholesky_identity():
    for n in range(MAXSIZE):
        x = torch.eye(n)
        _test_torch_cholesky(x)

def test_torch_cholesky_diagonal():
    for n in range(MAXSIZE):
        x = 2 * torch.eye(n)
        _test_torch_cholesky(x)

def test_torch_cholesky_diagonal_inc():
    for n in range(MAXSIZE):
        x = torch.eye(n)
        for i in range(n):
            x[i,i] = i + 2
        _test_torch_cholesky(x)

def test_torch_cholesky_diagonal_dec():
    for n in range(MAXSIZE):
        x = torch.eye(n)
        for i in range(n):
            x[i,i] = n + 2 - i
        _test_torch_cholesky(x)

def test_torch_cholesky_basic1():
    x = torch.tensor([[2.,-1.,0.],[-1.,3.,-1.],[0.,-1.,4.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic2():
    x = torch.tensor([[25.,15.,-5.],[15.,18.,0.],[-5.,0.,11.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic3():
    x = torch.tensor([[18.,22.,54.,42.],[22.,70.,86.,62.],[54.,86.,174.,134.],[42.,62.,134.,106.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic4():
    x = torch.tensor([[0.9744,1.1998,0.4542,0.6176,0.4425],
                      [1.1998,2.3837,1.1871,1.5825,1.8001],
                      [0.4542,1.1871,0.8640,0.6495,1.3102],
                      [0.6176,1.5825,0.6495,1.5022,1.3349],
                      [0.4425,1.8001,1.3102,1.3349,2.3573]])
    _test_torch_cholesky(x)

# Test random small matrices of random size between 1x1 and 10x10
def test_torch_cholesky_random_small():
    for _ in range(0,100):
        N = random.randint(1,10)
        x = torch.rand((N,N))
        x = torch.mm(x, x.t()) # make symmetric positive-semidefinite
        x.add_(torch.eye(N))   # make positive definite
        _test_torch_cholesky(x)

# Test random big matrices of random size between 100x100 and 1000x1000
def test_torch_cholesky_random_big():
    for _ in range(0,20):
        N = random.randint(100,1001)
        x = torch.rand((N,N))
        x = torch.mm(x, x.t()) # make symmetric positive-semidefinite
        x.add_(torch.eye(N))   # make positive definite
        _test_torch_cholesky(x)

# Test random matrices of random size between 1x1 and 100x100
def test_torch_cholesky_random1():
    for _ in range(0,100):
        N = random.randint(1,101)
        x = torch.rand((N,N))
        x = torch.mm(x, x.t()) # make symmetric positive-semidefinite
        x.add_(torch.eye(N))   # make positive definite
        _test_torch_cholesky(x)

# Test random matrices of all sizes from 1x1 to 100x100
def test_torch_cholesky_random2():
    for N in range(1,101):
        x = torch.rand((N,N))
        x = torch.mm(x, x.t()) # make symmetric positive-semidefinite
        x.add_(torch.eye(N))   # make positive definite
        _test_torch_cholesky(x)

def test_torch_cholesky_random_neg():
    for _ in range(0,100):
        N = random.randint(1,100)
        x = torch.rand((N,N)) - 0.5 # make half the numbers negative
        x = torch.mm(x, x.t()) # make symmetric positive-semidefinite
        x.add_(torch.eye(N))   # make positive definite
        _test_torch_cholesky(x, atol=1e-6)

@settings(deadline=None)
@given(inputs=hu.tensors2dsquare(min_shape=1, max_shape=5))
def test_torch_cholesky_hypothesis(inputs):
    x = torch.tensor(inputs)
    x = torch.mm(x, x.t()) # make symmetric positive-semidefinite
    x.add_(10 * torch.eye(inputs.shape[0])) # make positive definite
    _test_torch_cholesky(x)

