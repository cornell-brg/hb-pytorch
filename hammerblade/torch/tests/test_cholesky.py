"""
Unit tests for torch.cholesky kernel
10/07/2020 Kexin Zheng (kz73@cornell.edu)
"""

import torch
import random
import pytest
#from hypothesis import given, settings
#from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_cholesky(A, atol=1e-8):
    h = A.hammerblade()
    assert h.device == torch.device("hammerblade")
    L_h  = h.cholesky_hammerblade()

    # reconstruct L and L^T
    L = L_h.cpu()
    LT = L.t()

    # calculate L * L^T
    LLT = torch.mm(L, LT)

    print('A')
    print(A)
    print('L')
    print(L)
    print('LLT')
    print(LLT)
    assert torch.allclose(A, LLT, atol=atol)


def test_torch_cholesky_basic1():
    x = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic2():
    x = torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic3():
    x = torch.tensor([[2.,0.,0.],[0.,2.,0.],[0.,0.,2.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic4():
    x = torch.tensor([[2.,0.,0.],[0.,3.,0.],[0.,0.,4.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic5():
    x = torch.tensor([[3.,0.,0.,0.],[0.,3.,0.,0.],[0.,0.,3.,0.],[0.,0.,0.,3.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic6():
    x = torch.tensor([[2.,0.,0.,0.],[0.,3.,0.,0.],[0.,0.,4.,0.],[0.,0.,0.,5.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic7():
    x = torch.tensor([[2.,0.,0.,0.,0.],[0.,2.,0.,0.,0.],[0.,0.,2.,0.,0.],[0.,0.,0.,2.,0.],[0.,0.,0.,0.,2.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic8():
    x = torch.tensor([[2.,0.,0.,0.,0.],[0.,3.,0.,0.,0.],[0.,0.,4.,0.,0.],[0.,0.,0.,5.,0.],[0.,0.,0.,0.,6.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic9():
    x = torch.tensor([[2.,-1.,0.],[-1.,3.,-1.],[0.,-1.,4.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic10():
    x = torch.tensor([[25.,15.,-5.],[15.,18.,0.],[-5.,0.,11.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic11():
    x = torch.tensor([[18.,22.,54.,42.],[22.,70.,86.,62.],[54.,86.,174.,134.],[42.,62.,134.,106.]])
    _test_torch_cholesky(x)

def test_torch_cholesky_basic12():
    x = torch.tensor([[0.9744,1.1998,0.4542,0.6176,0.4425],
                      [1.1998,2.3837,1.1871,1.5825,1.8001],
                      [0.4542,1.1871,0.8640,0.6495,1.3102],
                      [0.6176,1.5825,0.6495,1.5022,1.3349],
                      [0.4425,1.8001,1.3102,1.3349,2.3573]])
    _test_torch_cholesky(x)

# Test random small matrices of random size between 1x1 and 100x100
def test_torch_cholesky_random_small():
    for _ in range(0,100):
        N = random.randint(2,10)
#        print('N', N)
        x = torch.rand((N,N))
        x = torch.mm(x, x.t()) # make symmetric positive-semidefinite
        x.add_(torch.eye(N))   # make positive definite
        _test_torch_cholesky(x, atol=1e-8)

# Test random matrices of random size between 1x1 and 100x100
def test_torch_cholesky_random1():
    for _ in range(0,50):
        N = random.randint(1,100)
        N = 22
        print('N', N)
        x = torch.rand((N,N))
        x = torch.mm(x, x.t()) # make symmetric positive-semidefinite
        x.add_(torch.eye(N))   # make positive definite
        print(x)
        _test_torch_cholesky(x, atol=1e-8)

# Test random matrices of all sizes from 1x1 to 100x100
def test_torch_cholesky_random2():
    for N in range(1,100):
#        print('N', N)
        x = torch.rand((N,N))
        x = torch.mm(x, x.t()) # make symmetric positive-semidefinite
        x.add_(torch.eye(N))   # make positive definite
        _test_torch_cholesky(x, atol=1e-8)

def test_torch_cholesky_random_neg():
    for _ in range(0,50):
        N = random.randint(1,100)
#        print('N', N)
        x = torch.rand((N,N)) - 0.5 # make half the numbers negative
        x = torch.mm(x, x.t()) # make symmetric positive-semidefinite
        x.add_(torch.eye(N))   # make positive definite
        _test_torch_cholesky(x, atol=1e-6)

'''
@settings(deadline=None)
@given(inputs=hu.tensors2dsquare(min_shape=1, max_shape=5))
def test_torch_cholesky_hypothesis(inputs):
    x = torch.tensor(inputs)
    x = torch.mm(x, x.t())
    assert x.dim() == 2
    _test_torch_cholesky(x, atol=1e-8)
'''

def main():
    #test_torch_cholesky_basic1()
    #test_torch_cholesky_basic2()
    #test_torch_cholesky_basic3()
    #test_torch_cholesky_basic4()
    #test_torch_cholesky_basic5()
    #test_torch_cholesky_basic6()
    #test_torch_cholesky_basic7()
    #test_torch_cholesky_basic8()
    #test_torch_cholesky_basic9()
    #test_torch_cholesky_basic10()
    #test_torch_cholesky_basic11()
    #test_torch_cholesky_basic12()
    #test_torch_cholesky_random_small()
    test_torch_cholesky_random1()
    test_torch_cholesky_random2()
    test_torch_cholesky_random_neg()

if __name__ == "__main__":
    main()
