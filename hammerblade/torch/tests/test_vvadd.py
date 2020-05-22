"""
Unit tests for torch.dot kernel
04/22/2020 Kexin Zheng (kz73@cornell.edu)
"""

import torch
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)

def _test_torch_vvadd(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    y_h = h1.vvadd(h2)
    print('original h1', h1.cpu())
    print('original h2', h2.cpu())
    print('sum', y_h.cpu())
    #assert y_h.device == torch.device("hammerblade")
    #assert torch.allclose(y_h.cpu(), x1)

def test_torch_vvadd_1():
    #x1 = torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.])
    #x2 = torch.tensor([9.,10.,11.,12.,13.,14.,15.,16.])
    x1 = torch.randn(96)
    x2 = torch.randn(96)
    _test_torch_vvadd(x1, x2)

