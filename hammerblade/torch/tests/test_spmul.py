"""
Unit tests for torch.cdist kernel
04/07/2021 Neil Adit (na469@cornell.edu)
"""
import torch 

torch.manual_seed(42)

def _test_torch_spmul_check(x1,x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    spmul_x = x1*x2
    spmul_h = h1*h2
    assert spmul_h.device == torch.device("hammerblade")
    assert torch.allclose(spmul_h.cpu().values(), spmul_x.values())

def test_torch_spmul_1():
    x1 = torch.rand(100,200).to_sparse().coalesce()
    x2_values = torch.rand(len(x1._values()))
    x2 = torch.sparse_coo_tensor(x1._indices(),x2_values,(100,200)).coalesce()
    _test_torch_spmul_check(x1,x2)

def test_torch_spmul_2():
    x1 = torch.rand(500,400).to_sparse().coalesce()
    x2_values = torch.rand(len(x1._values()))
    x2 = torch.sparse_coo_tensor(x1._indices(),x2_values,(500,400)).coalesce()
    _test_torch_spmul_check(x1,x2)

def test_torch_spmul_3():
    x1 = torch.rand(1000,200).to_sparse().coalesce()
    x2_values = torch.rand(len(x1._values()))
    x2 = torch.sparse_coo_tensor(x1._indices(),x2_values,(1000,200)).coalesce()
    _test_torch_spmul_check(x1,x2)