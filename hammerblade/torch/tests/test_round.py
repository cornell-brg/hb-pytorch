"""
Unit tests for torch.round kernel
04/12/2020 Andrew Pareles (amp342@cornell.edu)
"""
import torch

torch.manual_seed(42)

def _test_torch_round(x): #x is a torch.Tensor
    expected_tensor = x.round()
    
    h = x.hammerblade()
    got_hb = h.round()
    got_device = got_hb.device
    got_tensor = got_hb.cpu()

    print("tensor to round:\n", x)
    print("expected:\n", expected_tensor)
    print("got:\n", got_tensor)

    assert got_device == torch.device("hammerblade")
    assert torch.equal(got_tensor, expected_tensor)

def test_torch_round_1():
    x = torch.Tensor([-2.5, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 2.5, 3.5, 4.5, 5.5, 6.5]) 
    _test_torch_round(x)

def test_torch_round_2():
    x = torch.randn(2, 3)
    _test_torch_round(x)