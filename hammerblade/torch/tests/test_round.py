"""
Unit tests for torch.round kernel
04/12/2020 Andrew Pareles (amp342@cornell.edu)
"""
import torch

torch.manual_seed(42)

def _test_torch_round(x): #x is a torch.Tensor
    h = x.hammerblade()
    round_x = x.round()
    round_h = h.round()
    assert round_h.device == torch.device("hammerblade")
    # assert torch.equal(round_h.cpu(), round_x)

def test_torch_round_1():
    x = torch.Tensor([.5,2.2,3.4,4.4555,4.55])
    _test_torch_round(x)
    print(x)

def test_torch_round_2():
    pass
    # x = torch.randn(3, 4)
    # _test_torch_round(x)
