"""
Unit tests for torch.hammerblade.profiler.fallback
05/18/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn.functional as F
import random
import pytest

torch.manual_seed(42)
random.seed(42)

def test_typedefault_1():
    x = torch.tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
    cpu = torch.atan(x)
    torch.hammerblade.profiler.enable()
    torch.hammerblade.profiler.fallback.enable()
    hb = torch.atan(x.hammerblade())
    torch.hammerblade.profiler.fallback.disable()
    torch.hammerblade.profiler.disable()
    assert torch.allclose(hb.cpu(), cpu)

@pytest.mark.xfail
def test_typedefault_1F():
    x = tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
    cpu = torch.atan(x)
    torch.hammerblade.profiler.enable()
    hb = torch.atan(x.hammerblade())
    torch.hammerblade.profiler.disable()
    assert torch.allclose(hb.cpu(), cpu)

def test_native_1():
    x = torch.randn(2, 3)
    cpu = torch.cat((x, x, x), 0)
    h = x.hammerblade()
    torch.hammerblade.profiler.enable()
    torch.hammerblade.profiler.fallback.enable()
    hb = torch.cat((h, h, h), 0)
    torch.hammerblade.profiler.fallback.disable()
    torch.hammerblade.profiler.disable()
    assert torch.allclose(hb.cpu(), cpu)

@pytest.mark.xfail
def test_native_1F():
    x = torch.randn(2, 3)
    cpu = torch.cat((x, x, x), 0)
    h = x.hammerblade()
    torch.hammerblade.profiler.enable()
    hb = torch.cat((h, h, h), 0)
    torch.hammerblade.profiler.disable()
    assert torch.allclose(hb.cpu(), cpu)

def test_complex_1():
    input = torch.randn(3)
    target = torch.empty(3).random_(2)
    cpu_loss = F.binary_cross_entropy_with_logits(input, target)
    torch.hammerblade.profiler.enable()
    torch.hammerblade.profiler.fallback.enable()
    hb_loss = F.binary_cross_entropy_with_logits(input.hammerblade(), target.hammerblade())
    torch.hammerblade.profiler.fallback.disable()
    torch.hammerblade.profiler.disable()
    assert torch.allclose(hb_loss.cpu(), cpu_loss)

@pytest.mark.xfail
def test_complex_1F():
    input = torch.randn(3)
    target = torch.empty(3).random_(2)
    cpu_loss = F.binary_cross_entropy_with_logits(input, target)
    hb_loss = F.binary_cross_entropy_with_logits(input.hammerblade(), target.hammerblade())
    assert torch.allclose(hb_loss.cpu(), cpu_loss)
