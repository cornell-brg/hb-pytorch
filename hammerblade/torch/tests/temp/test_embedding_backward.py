"""
Tests on torch.nn.Embedding
04/22/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn as nn
import random
import copy

torch.manual_seed(42)
random.seed(42)

def _test_torch_nn_embedding_back(input, padding_idx=None):
    if padding_idx is not None:
        embedding = nn.Embedding(10, 3, padding_idx=padding_idx)
    else:
        embedding = nn.Embedding(10, 3)
    embedding_h = copy.deepcopy(embedding).hammerblade()
    out = embedding(input)
    out_h = embedding_h(input.hammerblade())
    assert out_h.device == torch.device("hammerblade")
    assert torch.equal(out, out_h.cpu())
    out.backward(out * -1.0)
    out_h.backward(out_h * -1.0)
    assert embedding.weight.grad is not None
    assert embedding_h.weight.grad is not None
    assert embedding_h.weight.grad.device == torch.device("hammerblade")
    assert torch.allclose(embedding.weight.grad, embedding_h.weight.grad.cpu())

def test_torch_nn_embedding_back_1():
    input = torch.LongTensor([1, 2, 4, 5])
    _test_torch_nn_embedding_back(input)
    _test_torch_nn_embedding_back(input, padding_idx=0)

def test_torch_nn_embedding_back_2():
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    _test_torch_nn_embedding_back(input)
    _test_torch_nn_embedding_back(input, padding_idx=0)

def test_torch_nn_embedding_back_3():
    input = torch.LongTensor([[0, 2, 0, 5], [4, 0, 0, 9]])
    _test_torch_nn_embedding_back(input)
    _test_torch_nn_embedding_back(input, padding_idx=0)
