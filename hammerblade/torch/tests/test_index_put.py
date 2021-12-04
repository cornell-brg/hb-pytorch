"""
Tests on torch.index
18/11/2021 Zhongyuan Zhao (zz546@cornell.edu)
"""
import torch

def test_index_put_1D_array():
    x = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2]).int()
    x1 = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2]).int()
    hb_x = x.hammerblade()
    hb_x1 = x1.hammerblade()

    #Index multiple elements using a 1D tensor as the indexer
    y = torch.tensor([[1], [4], [6]])
    value = torch.ones(y.shape[0]).int()
    x.index_put_(tuple(y.t()), value) #Output[2, 5, 7]
    x1.index_put_(tuple(y.t()), value, accumulate=True)

    hb_y = y.hammerblade()
    hb_value = value.hammerblade()
    hb_x.index_put_(tuple(hb_y.t()), hb_value)
    hb_x1.index_put_(tuple(hb_y.t()), hb_value, accumulate=True)
    assert torch.allclose(x, hb_x.cpu())
    assert torch.allclose(x1, hb_x1.cpu())

def test_index_put_2D_array():
    x = torch.tensor([[2, 2, 2, 2], [2, 2, 2, 2]]).int()
    x1 = torch.tensor([[2, 2, 2, 2], [2, 2, 2, 2]]).int()
    hb_x = x.hammerblade()
    hb_x1 = x1.hammerblade()

    #Index multiple elements using a 1D tensor as the indexer
    y = torch.tensor([[0, 0], [0, 2], [1, 1], [1, 3]])
    value = torch.ones(y.shape[0]).int()
    x.index_put_(tuple(y.t()), value) #Output[2, 5, 7]
    x1.index_put_(tuple(y.t()), value, accumulate=True)

    hb_y = y.hammerblade()
    hb_value = value.hammerblade()
    hb_x.index_put_(tuple(hb_y.t()), hb_value)
    hb_x1.index_put_(tuple(hb_y.t()), hb_value, accumulate=True)
    assert torch.allclose(x, hb_x.cpu())
    assert torch.allclose(x1, hb_x1.cpu())

