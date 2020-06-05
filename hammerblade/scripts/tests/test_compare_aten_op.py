import sys
import pathlib
current_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(current_path + '/..')
import std_parser
import stack_parser
from compare_aten_op import HB_FREQUENCY, compare

def test_compare_aten_op_1():

    aten_op = compare(current_path + "/demo/full.std", current_path + "/demo/chunk.std",
                      current_path + "/demo/manycore_stats.log")

    # pytest assertion
    assert aten_op.hb_device_time == 1.204603
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
    assert aten_op.hb_device_time == 1.204603
    assert aten_op.hb_host_time == 170.0
    assert aten_op.xeon_time == 34.451
