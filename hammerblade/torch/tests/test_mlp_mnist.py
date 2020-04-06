"""
Test on multilayer perceptron MNIST
04/06/2020 Lin Cheng (lc873@cornell.edu)
"""

import os
import copy
import torch
import torch.nn as nn
import numpy as np
import hbutils
import pytest

torch.manual_seed(42)

#-------------------------------------------------------------------------
# Multilayer Preception for MNIST
#-------------------------------------------------------------------------
# Dropout set to 0 so the backward is deterministic

class MLPModel(nn.Module):

  def __init__(self):
    super(MLPModel, self).__init__()

    self.mnist = nn.Sequential \
    (
      nn.Linear(784, 128),
      nn.ReLU(),
      nn.Dropout(0.0),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Dropout(0.0),
      nn.Linear(64, 10),
    )

  def forward(self, x):
    return self.mnist(x)

#-------------------------------------------------------------------------
# Forward pass
#-------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get('USE_HB_EMUL') is None, reason="Slow on cosim")
def test_inference():
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
  output_cpu = model_cpu(image.view(-1, 28*28))

  # inference on HammerBlade
  output_hb = model_hb(image_hb.view(-1, 28*28))

  # check outputs
  assert output_hb.device == torch.device("hammerblade")
  assert torch.allclose(output_cpu, output_hb.cpu(), atol=1e-06)
