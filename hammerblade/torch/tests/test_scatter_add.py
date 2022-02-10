"""
Tests on torch.scatter_add_
Feb/10/2021 Zhongyuan Zhao (zz546@cornell.edu)
"""
import torch

def test_scatter_1():
    src = torch.ones((2, 5))
    hb_src = src.hammerblade()
    index = torch.tensor([[0, 1, 2, 0, 0]])
    hb_index = index.hammerblade()
    result = torch.zeros(3, 5, dtype=src.dtype)
    hb_result = result.hammerblade()
    result = result.scatter_add_(0, index, src)
    hb_result = hb_result.scatter_add(0, hb_index, hb_src)

    print(result)
    print(hb_result)

    assert hb_result.device == torch.device("hammerblade")
    assert torch.allclose(result, hb_result.cpu())


def test_scatter_2():
    src = torch.ones((2, 5))
    hb_src = src.hammerblade()
    index = torch.tensor([[0, 1, 2, 0, 0]])
    hb_index = index.hammerblade()
    result = torch.zeros(3, 5, dtype=src.dtype)
    hb_result = result.hammerblade()
    result = result.scatter_add_(1, index, src)
    hb_result = hb_result.scatter_add(1, hb_index, hb_src)

    print(result)
    print(hb_result)

    assert hb_result.device == torch.device("hammerblade")
    assert torch.allclose(result, hb_result.cpu())

def test_scatter_3():
    src = torch.ones((2, 5))
    hb_src = src.hammerblade()
    index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
    hb_index = index.hammerblade()
    result = torch.zeros(3, 5, dtype=src.dtype)
    hb_result = result.hammerblade()
    result = result.scatter_add_(0, index, src)
    hb_result = hb_result.scatter_add(0, hb_index, hb_src)

    assert hb_result.device == torch.device("hammerblade")
    assert torch.allclose(result, hb_result.cpu())

def test_scatter_4():
    src = torch.ones((2, 5))
    hb_src = src.hammerblade()
    index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
    hb_index = index.hammerblade()
    result = torch.zeros(3, 5, dtype=src.dtype)
    hb_result = result.hammerblade()
    result = result.scatter_add_(1, index, src)
    hb_result = hb_result.scatter_add(1, hb_index, hb_src)

    assert hb_result.device == torch.device("hammerblade")
    assert torch.allclose(result, hb_result.cpu())
