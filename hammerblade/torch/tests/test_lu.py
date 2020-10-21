"""
Unit tests for torch.lu kernel
10/07/2020 Kexin Zheng (kz73@cornell.edu)
"""

import torch
import random
import pytest
#from hypothesis import given, settings
#from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_lu(x, atol=1e-8):
#    print("--------------input")
#    print(x)
    h = x.hammerblade()

    fac_h, piv_h = h.lu_hammerblade()
#    print("\nham factorization")
#    print(fac_h)
#    print("ham pivots")
#    print(piv_h)
#    piv_h = piv_h.hammerblade()

#    fac_c, piv_c, infos_c = x.lu(pivot=True, get_infos=True)
#    print("cpu factorization")
#    print(fac_c)
#    print("cpu pivots")
#    print(piv_c)
#    print("cpu infos")
#    print(infos_c)

    assert h.device == torch.device("hammerblade")

    # reconstruct L and U
    LU_h = fac_h.cpu()
    upper = torch.zeros_like(x)
    lower = torch.zeros_like(x)
    for i in range(0, x.size(0)):
        for j in range(0, x.size(1)):
            if i == j :
                lower[i][j] = 1
                upper[i][j] = LU_h[i][j]
            elif i < j:
                upper[i][j] = LU_h[i][j]
            else:
                lower[i][j] = LU_h[i][j]

    # reconstruct P matrix
    pivots = torch.zeros_like(x)
    for i in range(0, len(piv_h.cpu())):
        p = piv_h.cpu()[i]
        pivots[i][p-1] = 1

    # calculate P*A and L*U
    PA = torch.matmul(pivots, x)
    LU = torch.matmul(lower, upper)

    print('A')
    print(x)
    #print('P')
    #print(pivots)
    #print('PA')
    #print(PA)
    #print('LU')
    #print(LU)
    assert torch.allclose(PA, LU, atol=atol)


def test_torch_lu_basic1():
    x = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    _test_torch_lu(x)

def test_torch_lu_basic2():
    x = torch.tensor([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
    _test_torch_lu(x)

def test_torch_lu_basic3():
    x = torch.tensor([[0.,2.,0.],[1.,0.,0.],[0.,0.,1.]])
    _test_torch_lu(x)

def test_torch_lu_basic4():
    x = torch.tensor([[5.,6.,7.],[10.,20.,23.],[15.,50.,67.]])
    _test_torch_lu(x)

def test_torch_lu_basic5():
    x = torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
    _test_torch_lu(x)

def test_torch_lu_random1():
    for i in range(0,50):
        N = random.randint(1,100)
        print('N', N)
        x = torch.rand((N,N))
        _test_torch_lu(x, atol=1e-5)

def test_torch_lu_random2():
    for N in range(1,100):
        print('N', N)
        x = torch.rand((N,N))
        _test_torch_lu(x, atol=1e-6)

'''
@settings(deadline=None)
#@given(inputs=hu.tensors(n=1, min_dim=2, max_dim=2, min_value=1, max_value=10))
@given(inputs=hu.tensors2dsquare(min_shape=1, max_shape=15))
def test_torch_lu_hypothesis0110(inputs):
    #print('\n\n-----------------INPUTS')
    #print(inputs.shape)
    #print(inputs)
    x = torch.tensor(inputs)
    print('SIZE')
    print(x.size())
    _test_torch_lu(x, atol=1e-5)
'''

def main():
    test_torch_lu_basic1()
    test_torch_lu_basic2()
    test_torch_lu_basic3()
'''
    test_torch_lu_basic4()
    test_torch_lu_basic5()
    test_torch_lu_random1()
    test_torch_lu_random2()
'''

if __name__ == "__main__":
    main()
