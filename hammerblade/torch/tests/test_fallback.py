"""
Unit tests for torch.hammerblade.profiler.fallback
05/18/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn as nn
import random
import pytest

torch.manual_seed(42)
random.seed(42)

def test_typedefault_1():
    x = torch.tensor([0.2341, 0.2539, -0.6256, -0.6448])
    cpu = torch.atan(x)
    torch.hammerblade.profiler.enable()
    torch.hammerblade.profiler.fallback.enable()
    hb = torch.atan(x.hammerblade())
    torch.hammerblade.profiler.fallback.disable()
    torch.hammerblade.profiler.disable()
    assert torch.allclose(hb.cpu(), cpu)

@pytest.mark.xfail
def test_typedefault_1F():
    x = tensor([0.2341, 0.2539, -0.6256, -0.6448])
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
    unfold = nn.Unfold(kernel_size=(2, 3))
    input = torch.randn(2, 5, 3, 4)
    torch.hammerblade.profiler.enable()
    torch.hammerblade.profiler.fallback.enable()
    output = unfold(input)
    output_h = unfold(input.hammerblade())
    torch.hammerblade.profiler.fallback.disable()
    torch.hammerblade.profiler.disable()
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())

@pytest.mark.xfail
def test_complex_1F():
    unfold = nn.Unfold(kernel_size=(2, 3))
    input = torch.randn(2, 5, 3, 4)
    output = unfold(input)
    output_h = unfold(input.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())
