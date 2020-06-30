"""
Unit tests for sddmm kernel
06/30/2020 Andrew Pareles (amp342@cornell.edu)
"""
import torch

torch.manual_seed(42)


def _test_torch_sddmm(a, b, c):
    # expected_tensor = torch.sddmm(a, b, c)
    ah = a.hammerblade()
    bh = b.hammerblade()
    ch = c.hammerblade()
    got_hb = torch.sddmm(ah, bh, ch)
    got_device = got_hb.device
    got_tensor = got_hb.cpu()
    print(got_tensor)
    assert got_device == torch.device("hammerblade")
    # assert torch.equal(got_tensor, expected_tensor)


def test_torch_round_1():
    a = torch.Tensor([[1, 0], [0, 3]]).to_sparse()
    b = torch.Tensor([[5, 3], [1, 7]])
    c = torch.Tensor([[1, 2], [2, 1]])
    _test_torch_sddmm(a, b, c)

# def test_torch_round_2():
#     x = torch.randn(2, 3)
#     _test_torch_round(x)
