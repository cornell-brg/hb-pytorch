"""
Unit tests for sddmm kernel
06/30/2020 Andrew Pareles (amp342@cornell.edu)
"""
import torch

torch.manual_seed(42)


def sddmm_expected(a, b, c):
    outvals = torch.zeros(a._nnz())
    for k in range(a._nnz()):
        ai, aj = tuple(a._indices()[:, k].tolist())
        brow = b[ai, :]
        ccol = c[:, aj]
        outvals[k] = torch.dot(brow, ccol)
    return torch.sparse.FloatTensor(
        a._indices(),
        outvals,
        a.shape,
    ).to_dense()

def _test_torch_sddmm(a, b, c):
    expected_tensor = sddmm_expected(a, b, c)
    ah = a.hammerblade()
    bh = b.hammerblade()
    ch = c.hammerblade()
    got_hb = torch.sddmm(ah, bh, ch)
    got_device = got_hb.device
    got_tensor = got_hb.cpu()
    print(got_tensor)
    print(expected_tensor)
    assert got_device == torch.device("hammerblade")
    assert torch.equal(got_tensor, expected_tensor)


def test_torch_sddmm_1():
    a = torch.Tensor([[1, 0, 1], [0, 3, 0]]).to_sparse()
    b = torch.Tensor([[5, 3], [1, 7]])
    c = torch.Tensor([[1, 2, 1], [2, 1, 1]])
    _test_torch_sddmm(a, b, c)

def test_torch_sddmm_2():
    a = torch.Tensor([[0, 0], [0, 0]]).to_sparse()
    b = torch.Tensor([[5, 3], [1, 7]])
    c = torch.Tensor([[1, 2], [2, 1]])
    _test_torch_sddmm(a, b, c)

def test_torch_sddmm_3():
    a = torch.Tensor([[1, 1], [1, 1]]).to_sparse()
    b = torch.Tensor([[5, 3], [1, 7]])
    c = torch.Tensor([[1, 2], [2, 1]])
    _test_torch_sddmm(a, b, c)
