"""
Test on multilayer perceptron MNIST
04/06/2020 Lin Cheng (lc873@cornell.edu)
"""

import os
import copy
import torch
import torch.nn as nn
import random
import hbutils
import pytest

torch.manual_seed(42)
random.seed(42)

# -------------------------------------------------------------------------
# Multilayer Preception for MNIST
# -------------------------------------------------------------------------
# Dropout set to 0 so the backward is deterministic

class MLPModel(nn.Module):

    def __init__(self):
        super(MLPModel, self).__init__()

        self.mnist = nn.Sequential(nn.Linear(784, 128),
                                   nn.ReLU(),
                                   nn.Dropout(0.0),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Dropout(0.0),
                                   nn.Linear(64, 10))

    def forward(self, x):
        return self.mnist(x)

# -------------------------------------------------------------------------
# Forward pass
# -------------------------------------------------------------------------

@pytest.mark.skipif(torch.hb_emul_on, reason="Slow on cosim")
def test_mlp_inference():
    # create CPU model with random parameters
    model_cpu = MLPModel()

    # create HammerBlade model by deepcopying
    model_hb = copy.deepcopy(model_cpu)
    model_hb.to(torch.device("hammerblade"))

    # set both models to use eval mode
    model_cpu.eval()
    model_hb.eval()

    # random 28 * 28 image
    image = torch.randn(28, 28, requires_grad=True)
    image_hb = hbutils.init_hb_tensor(image)

    # inference on CPU
    output_cpu = model_cpu(image.view(-1, 28 * 28))

    # inference on HammerBlade
    output_hb = model_hb(image_hb.view(-1, 28 * 28))

    # check outputs
    assert output_hb.device == torch.device("hammerblade")
    assert torch.allclose(output_cpu, output_hb.cpu(), atol=1e-06)

# -------------------------------------------------------------------------
# Backward pass
# -------------------------------------------------------------------------

@pytest.mark.skipif(torch.hb_emul_on, reason="Slow on cosim")
def test_mlp_backprop():
    # create CPU model with random parameters
    model_cpu = MLPModel()

    # create HammerBlade model by deepcopying
    model_hb = copy.deepcopy(model_cpu)
    model_hb.to(torch.device("hammerblade"))

    # set both models to use train mode
    model_cpu.train()
    model_hb.train()

    # random 28 * 28 image
    image = torch.randn(28, 28, requires_grad=True)
    image_hb = hbutils.init_hb_tensor(image)

    # inference on CPU
    output_cpu = model_cpu(image.view(-1, 28 * 28))

    # inference on HammerBlade
    output_hb = model_hb(image_hb.view(-1, 28 * 28))

    # check outputs
    assert output_hb.device == torch.device("hammerblade")
    assert torch.allclose(output_cpu, output_hb.cpu(), atol=1e-06)

    # random gradients
    grad = torch.rand(output_cpu.shape)
    grad_hb = grad.hammerblade()

    # backprop
    output_cpu.backward(grad)
    output_hb.backward(grad_hb)

    # compare parameters (adapted from test_lenet5.py)
    named_parameters = reversed(list(model_cpu.named_parameters()))
    parameters_hb = reversed(list(model_hb.parameters()))
    for (name, param), param_hb in zip(named_parameters, parameters_hb):
        # iterate in reversed order so that this fails at earliest failure
        # during backprop
        assert torch.allclose(param.grad, param_hb.grad.cpu(), atol=1e-6), \
            name + " value mismatch"

    # Compare input gradients
    assert torch.allclose(image.grad, image_hb.grad.cpu(), atol=1e-6)
