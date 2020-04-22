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

def _test_torch_nn_embedding(input, padding_idx=None):
    if padding_idx is not None:
        embedding = nn.Embedding(10, 3, padding_idx=padding_idx)
    else:
        embedding = nn.Embedding(10, 3)
    embedding_h = copy.deepcopy(embedding).hammerblade()
    out = embedding(input)
    out_h = embedding_h(input.hammerblade())
    assert out_h.device == torch.device("hammerblade")
    assert torch.equal(out, out_h.cpu())

def test_torch_nn_embedding_1():
    input = torch.LongTensor([1,2,4,5])
    _test_torch_nn_embedding(input)
    _test_torch_nn_embedding(input, padding_idx=0)

def test_torch_nn_embedding_2():
    input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    _test_torch_nn_embedding(input)
    _test_torch_nn_embedding(input, padding_idx=0)

def test_torch_nn_embedding_3():
    input = torch.LongTensor([[0,2,0,5],[4,0,0,9]])
    _test_torch_nn_embedding(input)
    _test_torch_nn_embedding(input, padding_idx=0)
