"""
Unit tests for dense-sparseT matrix product (dstmm) kernel
06/30/2020 Andrew Pareles (amp342@cornell.edu)
"""
import torch

torch.manual_seed(42)

def _test_torch_dstmm(a, bT):
    expected_tensor = (a@bT.t().to_dense())
    ah = a.hammerblade()
    bTh = bT.hammerblade()
    got_hb = torch.dstmm(ah, bTh)
    got_device = got_hb.device
    got_tensor = got_hb.cpu()
    assert got_device == torch.device("hammerblade")
    assert torch.equal(got_tensor, expected_tensor)
    # compare with CPU
    expected_tensor_cpu = torch.dstmm(a, bT)
    assert torch.equal(got_tensor, expected_tensor_cpu)


def test_torch_dstmm_1():
    a = torch.Tensor([[1, 2, 3]])
    b = torch.Tensor([[0, 0, 0, 4], [0, 1, 2, 0], [0, 0, 0, 1]]).t().to_sparse()
    _test_torch_dstmm(a, b)

def test_torch_dstmm_2():
    a = torch.Tensor([[0, 1], [1, 0]])
    b = torch.Tensor([[5, 0], [1, 0]]).t().to_sparse()
    _test_torch_dstmm(a, b)

def test_torch_dstmm_3():
    a = torch.Tensor([[1, 2], [1, 1]])
    b = torch.Tensor([[5, 3], [1, 7]]).t().to_sparse()
    _test_torch_dstmm(a, b)
