import torch

def test_bitwise_not_bool():
    a = torch.tensor([0, 1, 1], dtype=torch.bool)
    b = torch.bitwise_not(a)
    hb_a = a.hammerblade()
    hb_b = torch.bitwise_not(hb_a)
    assert hb_b.device == torch.device("hammerblade")
    assert torch.equal(hb_b.cpu(), b)

def test_bitwise_not_int():
    a = torch.tensor([0, 1, 1], dtype=torch.int32)
    b = torch.bitwise_not(a)
    hb_a = a.hammerblade()
    hb_b = torch.bitwise_not(hb_a)
    assert hb_b.device == torch.device("hammerblade")
    assert torch.allclose(hb_b.cpu(), b)
