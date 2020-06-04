import sys
import os
sys.path.append(os.getcwd() + '/..')
import stack_parser

def test_stack_parsing_1():
    raw_stack = """
#TOP_LEVEL_FUNC#__at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar);0.399
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@CPU_LOG@;0.067
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@CPU_LOG@<|>at::Tensor at::CPUType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>);0.015
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@CPU_LOG@<|>at::native::add_stub::add_stub();0.009
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@;0.234
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::Tensor at::HammerBladeType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>);0.01
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::native::add_stub::add_stub();0.187
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::native::add_stub::add_stub()<|>@OFFLOAD_KERNEL@__tensorlib_add;0.145
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::native::add_stub::add_stub()<|>@OFFLOAD_KERNEL@__tensorlib_add<|>@TRIM@;0
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>at::Tensor at::CPUType::{anonymous}::llcopy(const at::Tensor&);0.025

#TOP_LEVEL_FUNC_END#__at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)
"""

    root = stack_parser.exec_time_tree(raw_stack)
    graph = stack_parser.exec_time_print_tree(root)
    assert graph == """|- Node(time_in_roi : 0.399)\n  |- Node(at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar) : 0.399)\n    |- Node(@CPU_LOG@ : 0.067)\n      |- Node(at::Tensor at::CPUType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>) : 0.015)\n      |- Node(at::native::add_stub::add_stub() : 0.009)\n    |- Node(@HB_LOG@ : 0.234)\n      |- Node(at::Tensor at::HammerBladeType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>) : 0.01)\n      |- Node(at::native::add_stub::add_stub() : 0.187)\n        |- Node(@OFFLOAD_KERNEL@__tensorlib_add : 0.145)\n          |- Node(@TRIM@ : 0.0)\n    |- Node(at::Tensor at::CPUType::{anonymous}::llcopy(const at::Tensor&) : 0.025)\n  |- Node(other : 0.0)"""

