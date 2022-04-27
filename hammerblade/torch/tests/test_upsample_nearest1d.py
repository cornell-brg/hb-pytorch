"""
Unit tests for torch.upsample_nearest1d kernel
 4/26/2022 Aditi Agarwal (aa2224@cornell.edu)
"""

import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_upsample_nearest1d(x,scale_factor):
    h = x.hammerblade()
    ans_x = x.upsample_nearest1d()