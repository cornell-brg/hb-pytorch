"""
Unit tests for validating sddtmm kernel
08/13/2020 Andrew Pareles (amp342@cornell.edu)
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

def _test_torch_sddtmm(a, b, c):
    expected_tensor = sddmm_expected(a, b, c.t())
    ah = a.hammerblade()
    bh = b.hammerblade()
    ch = c.hammerblade()
    got_hb = torch.sddtmm(ah, bh, ch)
    got_device = got_hb.device
    got_tensor = got_hb.cpu()
    assert got_device == torch.device("hammerblade")
    assert torch.equal(got_tensor, expected_tensor)
    # compare with CPU
    expected_tensor_cpu = torch.sddtmm(a, b, c)
    assert torch.equal(got_tensor, expected_tensor_cpu)


def test_torch_sddtmm_1():
    a = torch.Tensor([[1, 0], [0, 3], [1, 0]]).to_sparse()
    b = torch.Tensor([[5, 3], [1, 7]])
    c = torch.Tensor([[1, 2], [2, 1], [1, 1]])
    _test_torch_sddtmm(a, b, c)

def test_torch_sddtmm_2():
    a = torch.Tensor([[0, 0], [0, 0]]).to_sparse()
    b = torch.Tensor([[5, 3], [1, 7]])
    c = torch.Tensor([[1, 2], [2, 1]])
    _test_torch_sddtmm(a, b, c)

def test_torch_sddtmm_3():
    a = torch.Tensor([[1, 0, 2], [0, 1, 0]]).to_sparse()
    b = torch.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    c = torch.Tensor([[1, 2, 3], [1, 1, 2]])
    _test_torch_sddtmm(a, b, c)
