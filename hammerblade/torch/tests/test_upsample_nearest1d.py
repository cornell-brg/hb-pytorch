"""
Unit tests for torch.upsample_nearest1d kernel
 4/26/2022 Aditi Agarwal (aa2224@cornell.edu)
"""

import torch
from torch import nn
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_upsample_nearest1d(x,scale_factor):
    h = x.hammerblade()
    ans_x = nn.Upsample(scale_factor = scale_factor)
    ans_h = nn.Upsample(scale_factor = scale_factor)
    assert h.device == torch.device("hammerblade")
    assert torch.equal(h.cpu(), ans_x)

def test_torch_upsample_1():
    x = torch.ones(10)
    _test_torch_upsample_nearest1d(x,5)

def test_torch_upsample_1():
    x = torch.randn(5)
    _test_torch_upsample_nearest1d(x,10)
