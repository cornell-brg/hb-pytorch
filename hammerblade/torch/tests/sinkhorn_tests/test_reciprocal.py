"""
Unit tests for validating reciprocal kernel (HB, sparseHB, sparseCPU)
08/28/2020 Andrew Pareles (amp342@cornell.edu)
"""
import torch

torch.manual_seed(42)

def _test_torch_reciprocal(a):
    expected_tensor = 1.0 / a.to_dense()
    # I'm not sure if a.hammerblade() creates a copy of the tensor,
    # but don't want to risk it:
    a1 = a
    a2 = a.clone()
    a3 = a.clone().to_dense()
    # sparseHB == expected
    ah1 = a1.hammerblade()
    got_hb1 = torch.reciprocal_(ah1)
    got_device1 = got_hb1.device
    got_tensor1 = got_hb1.cpu()
    print("expected", expected_tensor)
    print("sparseHB", got_tensor1.to_dense())
    assert got_device1 == torch.device("hammerblade")
    assert torch.equal(got_tensor1.to_dense(), expected_tensor)
    # sparseHB == sparseCPU
    expected_tensor_cpu = torch.reciprocal_(a2)
    print("sparseCPU", expected_tensor_cpu.to_dense() )
    assert torch.equal(got_tensor1.to_dense(), expected_tensor_cpu.to_dense())
    # sparseHB == HB
    ah3 = a3.hammerblade()
    got_hb3 = torch.reciprocal_(ah3)
    got_device3 = got_hb3.device
    got_tensor3 = got_hb3.cpu()
    print("HB", expected_tensor_cpu.to_dense() )
    assert got_device3 == torch.device("hammerblade")
    assert torch.equal(got_tensor1.to_dense(), got_tensor3)

def test_torch_reciprocal_1():
    a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]]).to_sparse()
    _test_torch_reciprocal(a)

def test_torch_reciprocal_2():
    a = torch.FloatTensor([[1, 0, 3], [4, 0, 6]]).to_sparse()
    _test_torch_reciprocal(a)

