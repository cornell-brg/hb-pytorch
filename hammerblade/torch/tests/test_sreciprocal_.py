"""
Unit tests for validating sreciprocal_ kernel 
08/28/2020 Andrew Pareles (amp342@cornell.edu)
"""
import torch

torch.manual_seed(42)

def _test_torch_sreciprocal_(a):
    # I'm not sure if a.hammerblade() creates a copy
    # of the tensor, but don't want to risk it:
    a1 = a
    a2 = a.clone()
    ah = a1.hammerblade()
    got_hb = torch.sreciprocal_(ah)
    got_device = got_hb.device
    got_tensor = got_hb.cpu()
    # HB == CPU
    expected_tensor_cpu = torch.sreciprocal_(a2)
    assert torch.equal(got_tensor.to_dense(), expected_tensor_cpu.to_dense())

def test_torch_sreciprocal_1():
    a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]]).to_sparse()
    _test_torch_sreciprocal_(a)

def test_torch_sreciprocal_2():
    a = torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]).to_sparse().coalesce()
    _test_torch_sreciprocal_(a)

def test_torch_sreciprocal_3():
    a = torch.FloatTensor([[1, 0, 2], [1e-307, 3, 1e-309]]).to_sparse().coalesce()
    _test_torch_sreciprocal_(a)
