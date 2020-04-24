"""
Unit tests for torch.dot kernel
04/22/2020 Kexin Zheng (kz73@cornell.edu)
"""

import torch
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)

def _test_torch_dummy(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    y_h = h1.dummy(h2)
    #assert y_h.device == torch.device("hammerblade")
    #assert torch.allclose(y_h.cpu(), x1)

def test_torch_dummy_1():
    x1 = torch.randn(96)
    x2 = torch.randn(96)
    _test_torch_dummy(x1, x2)

