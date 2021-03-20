"""
Unit tests for torch.nn.F.binary_cross_entropy_with_logits
06/16/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn.functional as F
import random
import hbutils

torch.manual_seed(42)
random.seed(42)

def test_torch_nn_F_BinaryCrossEntropyWithLogits_1():
    input = torch.randn(3, requires_grad=True)
    input_h = hbutils.init_hb_tensor(input)
    assert input_h is not input
    target = torch.empty(3).random_(2)
    output = F.binary_cross_entropy_with_logits(input, target)
    output_h = F.binary_cross_entropy_with_logits(input_h, target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())
    output.backward()
    output_h.backward()
    assert input.grad is not None
    assert input_h.grad is not None
    assert input.grad is not input_h.grad
    assert torch.allclose(input.grad, input_h.grad.cpu())
