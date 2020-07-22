"""
Unit tests for dense-sparse matrix product (dsmp) kernel
06/30/2020 Andrew Pareles (amp342@cornell.edu)
"""
import torch

torch.manual_seed(42)

def _test_torch_dsmm(a, bT):
    expected_tensor = (a@bT.t().to_dense())
    print(expected_tensor)
    ah = a.hammerblade()
    bTh = bT.hammerblade()
    got_hb = torch.dstmp(ah, bTh)
    got_device = got_hb.device
    got_tensor = got_hb.cpu()
    print(got_tensor)
    print(expected_tensor)
    assert got_device == torch.device("hammerblade")
    assert torch.equal(got_tensor, expected_tensor)


def test_torch_dsmm_1():
    a = torch.Tensor([[1, 2, 3]])
    b = torch.Tensor([[0, 0, 0, 4], [0, 1, 2, 0], [0, 0, 0, 1]]).t().to_sparse()
    _test_torch_dsmm(a, b)

def test_torch_dsmm_2():
    a = torch.Tensor([[0, 1], [1, 0]])
    b = torch.Tensor([[5, 0], [1, 0]]).t().to_sparse()
    _test_torch_dsmm(a, b)

def test_torch_dsmm_3():
    a = torch.Tensor([[1, 2], [1, 1]])
    b = torch.Tensor([[5, 3], [1, 7]]).t().to_sparse()
    _test_torch_dsmm(a, b)
