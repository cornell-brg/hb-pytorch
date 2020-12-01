import torch
import torch.nn as nn
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

def test_torch_ss_1():
    x = torch.rand(10)
    _test_torch_ss_check(x,2)

def test_torch_ss_2():
    x = torch.randn(10)
    _test_torch_ss_check(x)

def test_torch_ss_3():
    x = torch.randn(100)
    _test_torch_ss_check(x,2,3)

def test_torch_ss_4():
    x = torch.randn(100)
    _test_torch_ss_check(x,10,3)

def test_torch_ss_5():
    x = torch.randn(1000)
    _test_torch_ss_check(x,10,3)

def test_torch_ss_6():
    x = torch.randn(10000)
    _test_torch_ss_check(x,15,3)

def test_torch_ss_7():
    x = torch.randn(100000)
    _test_torch_ss_check(x,16,5)

def _test_torch_ss_check(tensor_self,nproc=1,sr=1):
    tensor_self_hb = torch.tensor(tensor_self).hammerblade()
    result_hb = torch.sample_sort(tensor_self_hb,nproc,sr)
    assert result_hb.device == torch.device("hammerblade")
    
    a = torch.tensor(tensor_self)
    a, indices = torch.sort(a)
    print(a,result_hb.cpu())

    assert torch.allclose(result_hb.cpu(),a.cpu())
    # assert torch.allclose(result_hb.cpu(), torch.sample_sort(torch.tensor(tensor_self)))
