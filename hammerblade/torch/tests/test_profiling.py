"""
Unit tests for torch.hammerblade.profiler
05/06/2020 Lin Cheng (lc873@cornell.edu)
"""

import json
import torch
import random

torch.manual_seed(42)
random.seed(42)
torch.hammerblade.init()

def test_ROI():
    assert not torch.hammerblade.profiler.is_in_ROI()
    torch.hammerblade.profiler.enable()
    assert torch.hammerblade.profiler.is_in_ROI()
    torch.hammerblade.profiler.disable()
    assert not torch.hammerblade.profiler.is_in_ROI()

def test_ROI_2():
    assert not torch.hammerblade.profiler.is_in_ROI()
    torch.hammerblade.profiler.enable()
    assert torch.hammerblade.profiler.is_in_ROI()
    torch.hammerblade.profiler.disable()
    assert not torch.hammerblade.profiler.is_in_ROI()

def test_execution_time_1():
    x = torch.ones(100000)
    torch.hammerblade.profiler.enable()
    x = torch.randn(100000)
    y = x + x
    torch.hammerblade.profiler.disable()
    fancy = torch.hammerblade.profiler.exec_time.fancy_print()
    assert fancy.find("aten::randn") != -1
    assert fancy.find("aten::add") != -1
    assert fancy.find("aten::ones") == -1

def test_execution_time_2():
    x = torch.ones(100000)
    torch.hammerblade.profiler.enable()
    x = torch.randn(100000)
    y = x + x
    torch.hammerblade.profiler.disable()
    stack = torch.hammerblade.profiler.exec_time.raw_stack()
    assert stack.find("at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)") != -1
    assert stack.find("at::Tensor at::TypeDefault::randn(c10::IntArrayRef, const c10::TensorOptions&)") != -1
    assert stack.find("at::Tensor at::TypeDefault::ones(c10::IntArrayRef, const c10::TensorOptions&)") == -1
    assert stack.find("at::Tensor& at::native::legacy::cpu::_th_normal_(at::Tensor&, double, double, at::Generator*)") != -1
    assert stack.find("at::native::add_stub::add_stub()") != -1

def test_unimpl_1():
    x = torch.ones(100000)
    torch.hammerblade.profiler.enable()
    x = torch.randn(100000)
    y = x + x
    torch.hammerblade.profiler.disable()
    unimpl = torch.hammerblade.profiler.unimpl.fancy_print()
    assert unimpl.find("aten::normal_") != -1

def test_unimpl_2():
    x = torch.ones(100000)
    x = torch.randn(100000)
    torch.hammerblade.profiler.enable()
    y = x + x
    torch.hammerblade.profiler.disable()
    unimpl = torch.hammerblade.profiler.unimpl.fancy_print()
    assert unimpl.find("aten::normal_") == -1

def test_chart_1():
    M = torch.randn(2, 3)
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    torch.hammerblade.profiler.chart.clear()
    torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)")
    torch.hammerblade.profiler.enable()
    torch.add(M, mat1)
    torch.hammerblade.profiler.disable()
    torch.hammerblade.profiler.chart.clear()
    chart = torch.hammerblade.profiler.chart.json()
    assert chart == "[]\n"

def test_chart_2():
    M = torch.randn(2, 3)
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    torch.hammerblade.profiler.chart.clear()
    torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)")
    torch.hammerblade.profiler.enable()
    torch.add(M, mat1)
    torch.addmm(M, mat1, mat2)
    torch.addmm(M, mat1, mat2)
    torch.hammerblade.profiler.disable()
    torch.hammerblade.profiler.chart.clear()
    chart = torch.hammerblade.profiler.chart.json()
    golden = """[
    {
        "offload": false,
        "signature": "at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)"
    },
    {
        "offload": false,
        "signature": "at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)"
    }
]
"""
    assert chart == golden

def test_route_1():
    M = torch.randn(2, 3)
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    route = """[
    {
        "offload": true,
        "signature": "at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)"
    },
    {
        "offload": true,
        "signature": "at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)"
    }
]
"""
    data = json.loads(route)
    torch.hammerblade.profiler.route.set_route_from_json(data)
    _route = torch.hammerblade.profiler.route.json()
    assert _route == route
    out1 = torch.addmm(M, mat1, mat2)
    out1 = out1 + M
    torch.hb_profiler.enable()
    out2 = torch.addmm(M, mat1, mat2)
    out2 = out2 + M
    torch.hb_profiler.disable()
    assert torch.allclose(out1, out2)
