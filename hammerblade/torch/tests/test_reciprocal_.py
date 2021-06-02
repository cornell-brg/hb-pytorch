"""
Unit tests for validating reciprocal kernel
08/28/2020 Andrew Pareles (amp342@cornell.edu)
"""
import torch

torch.manual_seed(42)

def _test_torch_reciprocal_(a):
    expected_tensor = 1.0 / a
    # I'm not sure if a.hammerblade() creates a copy
    # of the tensor, but don't want to risk it:
    a1 = a
    a2 = a.clone()
    ah1 = a1.hammerblade()
    got_hb = torch.reciprocal_(ah1)
    got_device = got_hb.device
    got_tensor = got_hb.cpu()
    # HB == expected
    assert got_device == torch.device("hammerblade")
    assert torch.equal(got_tensor, expected_tensor)
    # HB == CPU
    expected_tensor_cpu = torch.reciprocal_(a2)
    assert torch.equal(got_tensor, expected_tensor_cpu)

def test_torch_reciprocal_1():
    a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    _test_torch_reciprocal_(a)

def test_torch_reciprocal_2():
    a = torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    _test_torch_reciprocal_(a)

def test_torch_reciprocal_3():
    a = torch.FloatTensor([[1, 0, 2], [1e-307, 3, 1e-309]])
    _test_torch_reciprocal_(a)
