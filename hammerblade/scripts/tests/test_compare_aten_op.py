import sys
import pathlib
current_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(current_path + '/..')
import std_parser
import stack_parser
from compare_aten_op import HB_FREQUENCY, compare, add_tree, draw_tree, average_aten_op

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

def test_compare_aten_op_1():

    aten_op = compare(current_path + "/demo/full.std", current_path + "/demo/chunk.std",
                      current_path + "/demo/manycore_stats.log")

    # pytest assertion
    assert aten_op.hb_device_time == 1204.603
    assert aten_op.hb_host_time == 170.0
    assert aten_op.xeon_time == 34.451
    cpu_graph = stack_parser.exec_time_print_tree(aten_op.cpu_log)
    assert cpu_graph == "|- Node(@CPU_LOG@ : 34.451)\n  |- Node(at::Tensor at::CPUType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>) : 0.043)\n  |- Node(at::native::add_stub::add_stub() : 34.316)"
    hb_graph = stack_parser.exec_time_print_tree(aten_op.hb_log)
    assert hb_graph == "|- Node(@HB_LOG@ : 170.0)\n  |- Node(at::Tensor at::HammerBladeType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>) : 0.004)\n  |- Node(at::native::add_stub::add_stub() : 170.0)\n    |- Node(@OFFLOAD_KERNEL@__tensorlib_add : 0.0)\n      |- Node(@TRIM@ : 0.0)"

def test_compare_aten_op_2():

    aten_op = compare(current_path + "/demo/full.std", current_path + "/demo/chunk.std",
                      current_path + "/demo/manycore_stats.log", fancy_func=True)

    # pytest assertion
    assert aten_op.hb_device_time == 1204.603
    assert aten_op.hb_host_time == 170.0
    assert aten_op.xeon_time == 34.451

def test_add_tree_1():
    root1 = stack_parser.exec_time_tree(raw_stack)
    root2 = stack_parser.exec_time_tree(raw_stack)
    sum_tree = add_tree(root1, root2)
    graph1 = draw_tree(root1)
    graph2 = draw_tree(root2)
    graph = draw_tree(sum_tree)
    assert graph1 == graph2
    assert graph1 == """
digraph {
0 [ shape=record label = "aten::time_in_roi|0.40"];
1 [ shape=record label = "aten::add|0.40"];
2 [ shape=record label = "@CPU_LOG@|0.07"];
3 [ shape=record label = "aten::empty|0.01"];
4 [ shape=record label = "aten::add_stub|0.01"];
5 [ shape=record label = "@HB_LOG@|0.23"];
6 [ shape=record label = "aten::empty|0.01"];
7 [ shape=record label = "aten::add_stub|0.19"];
8 [ shape=record label = "@OFFLOAD_KERNEL@__tensorlib_add|0.14"];
10 [ shape=record label = "aten::llcopy|0.03"];
11 [ shape=record label = "aten::other|0.00"];
0 -> 1;
0 -> 11;
1 -> 2;
1 -> 5;
1 -> 10;
2 -> 3;
2 -> 4;
5 -> 6;
5 -> 7;
7 -> 8;
}
"""
    assert graph == """
digraph {
0 [ shape=record label = "aten::time_in_roi|0.80"];
1 [ shape=record label = "aten::add|0.80"];
2 [ shape=record label = "@CPU_LOG@|0.13"];
3 [ shape=record label = "aten::empty|0.03"];
4 [ shape=record label = "aten::add_stub|0.02"];
5 [ shape=record label = "@HB_LOG@|0.47"];
6 [ shape=record label = "aten::empty|0.02"];
7 [ shape=record label = "aten::add_stub|0.37"];
8 [ shape=record label = "@OFFLOAD_KERNEL@__tensorlib_add|0.29"];
10 [ shape=record label = "aten::llcopy|0.05"];
11 [ shape=record label = "aten::other|0.00"];
0 -> 1;
0 -> 11;
1 -> 2;
1 -> 5;
1 -> 10;
2 -> 3;
2 -> 4;
5 -> 6;
5 -> 7;
7 -> 8;
}
"""

def test_average_aten_op_1():

    aten_op1 = compare(current_path + "/demo/full.std", current_path + "/demo/chunk.std",
                      current_path + "/demo/manycore_stats.log")
    aten_op2 = compare(current_path + "/demo/full.std", current_path + "/demo/chunk.std",
                      current_path + "/demo/manycore_stats.log")

    aten_op = average_aten_op([aten_op1, aten_op2])

    assert aten_op.hb_device_time == aten_op1.hb_device_time
    assert aten_op.hb_host_time == aten_op1.hb_host_time
    assert aten_op.xeon_time == aten_op1.xeon_time
    assert aten_op.hb_device_time == 1204.603
    assert aten_op.hb_host_time == 170.0
    assert aten_op.xeon_time == 34.451
    cpu_graph = stack_parser.exec_time_print_tree(aten_op.cpu_log)
    assert cpu_graph == "|- Node(@CPU_LOG@ : 34.451)\n  |- Node(at::Tensor at::CPUType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>) : 0.043)\n  |- Node(at::native::add_stub::add_stub() : 34.316)"
    hb_graph = stack_parser.exec_time_print_tree(aten_op.hb_log)
    assert hb_graph == "|- Node(@HB_LOG@ : 170.0)\n  |- Node(at::Tensor at::HammerBladeType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>) : 0.004)\n  |- Node(at::native::add_stub::add_stub() : 170.0)\n    |- Node(@OFFLOAD_KERNEL@__tensorlib_add : 0.0)\n      |- Node(@TRIM@ : 0.0)"
