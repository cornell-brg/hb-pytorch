"""
Unit tests for torch.cdist kernel
02/17/2021 Neil Adit (na469@cornell.edu)
"""
import torch 

torch.manual_seed(42)

def _test_torch_cdist_check(x1,x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    cdist_x = torch.cdist(x1,x2)
    cdist_h = torch.cdist(h1,h2)
    assert cdist_h.device == torch.device("hammerblade")
    assert torch.allclose(cdist_h.cpu(), cdist_x)

def test_torch_cdist_1():
    x1 = torch.randn(20,100)
    x2 = torch.randn(30,100)
    _test_torch_cdist_check(x1,x2)

def test_torch_cdist_2():
    x1 = torch.randn(20,1000)
    x2 = torch.randn(30,1000)
    _test_torch_cdist_check(x1,x2)

def test_torch_cdist_3():
    x1 = torch.randn(20,10)
    x2 = torch.randn(30,100)
    _test_torch_cdist_check(x1,x2)