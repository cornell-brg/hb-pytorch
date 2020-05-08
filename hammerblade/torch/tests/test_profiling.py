"""
Unit tests for torch.hb_profiler
05/06/2020 Lin Cheng (lc873@cornell.edu)
"""

import json
import torch
import random
import pytest

torch.manual_seed(42)
random.seed(42)

def test_ROI():
    assert not torch.hb_profiler.is_in_ROI()
    torch.hb_profiler.enable()
    assert torch.hb_profiler.is_in_ROI()
    torch.hb_profiler.disable()
    assert not torch.hb_profiler.is_in_ROI()

def test_ROI_2():
    assert not torch.hb_profiler.is_in_ROI()
    torch.hb_profiler.enable()
    assert torch.hb_profiler.is_in_ROI()
    torch.hb_profiler.disable()
    assert not torch.hb_profiler.is_in_ROI()

def test_execution_time_1():
    x = torch.ones(100000)
    torch.hb_profiler.enable()
    x = torch.randn(100000)
    y = x + x
    torch.hb_profiler.disable()
    fancy = torch.hb_profiler.exec_time.fancy_print()
    assert fancy.find("aten::randn") != -1
    assert fancy.find("aten::add") != -1
    assert fancy.find("aten::ones") == -1

def test_execution_time_2():
    x = torch.ones(100000)
    torch.hb_profiler.enable()
    x = torch.randn(100000)
    y = x + x
    torch.hb_profiler.disable()
    stack = torch.hb_profiler.exec_time.raw_stack()
    assert stack.find("at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)") != -1
    assert stack.find("at::Tensor at::TypeDefault::randn(c10::IntArrayRef, const c10::TensorOptions&)") != -1
    assert stack.find("at::Tensor at::TypeDefault::ones(c10::IntArrayRef, const c10::TensorOptions&)") == -1
    assert stack.find("at::Tensor& at::native::legacy::cpu::_th_normal_(at::Tensor&, double, double, at::Generator*)") != -1
    assert stack.find("at::native::add_stub::add_stub()") != -1

def test_unimpl_1():
    x = torch.ones(100000)
    torch.hb_profiler.enable()
    x = torch.randn(100000)
    y = x + x
    torch.hb_profiler.disable()
    unimpl = torch.hb_profiler.unimpl.print()
    assert unimpl.find("aten::normal_") != -1

def test_unimpl_2():
    x = torch.ones(100000)
    x = torch.randn(100000)
    torch.hb_profiler.enable()
    y = x + x
    torch.hb_profiler.disable()
    unimpl = torch.hb_profiler.unimpl.print()
    assert unimpl.find("aten::normal_") == -1

def test_chart_1():
    M = torch.randn(2, 3)
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    torch.hb_profiler.chart.clear_beacon()
    torch.hb_profiler.chart.add_beacon("at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)")
    torch.hb_profiler.enable()
    torch.add(M, mat1)
    torch.hb_profiler.disable()
    torch.hb_profiler.chart.clear_beacon()
    chart = torch.hb_profiler.chart.print()
    assert chart == "[]\n"

def test_chart_2():
    M = torch.randn(2, 3)
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    torch.hb_profiler.chart.clear_beacon()
    torch.hb_profiler.chart.add_beacon("at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)")
    torch.hb_profiler.enable()
    torch.add(M, mat1)
    torch.addmm(M, mat1, mat2)
    torch.addmm(M, mat1, mat2)
    torch.hb_profiler.disable()
    torch.hb_profiler.chart.clear_beacon()
    chart = torch.hb_profiler.chart.print()
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
        "offload": false,
        "signature": "at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)"
    },
    {
        "offload": false,
        "signature": "at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)"
    }
]
"""
    data = json.loads(route)
    torch.hb_profiler.route.set_route_from_json(data)
    _route = torch.hb_profiler.route.print()
    assert _route == route
    torch.hb_profiler.enable()
    torch.addmm(M, mat1, mat2)
    torch.add(M, mat1)
    torch.hb_profiler.disable()

@pytest.mark.xfail
def test_route_2_F():
    M = torch.randn(2, 3)
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    # TODO: reset route
    route = """[
    {
        "offload": false,
        "signature": "at::Tensor at::CPUType::{anonymous}::addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::Scalar, c10::Scalar)"
    },
    {
        "offload": false,
        "signature": "at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)"
    }
]
"""
    data = json.loads(route)
    torch.hb_profiler.route.set_route_from_json(data)
    torch.hb_profiler.enable()
    torch.addmm(M, mat1, mat2)
    torch.addmm(M, mat1, mat2)
    torch.hb_profiler.disable()
