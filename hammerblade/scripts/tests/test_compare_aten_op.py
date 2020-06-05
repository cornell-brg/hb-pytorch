import sys
import pathlib
current_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(current_path + '/..')
import std_parser
import stack_parser
from compare_aten_op import HB_FREQUENCY, compare

def test_compare_aten_op_1():
    # read actuals and stacks
    full_actuals = None
    full_raw_stack = None
    chunk_actuals = None
    chunk_raw_stack = None

    with open(current_path + "/demo/full.std", "r") as f_full:
        full_actuals, full_raw_stack = std_parser.parse(f_full.read())
    with open(current_path + "/demo/chunk.std", "r") as f_chunk:
        chunk_actuals, chunk_raw_stack = std_parser.parse(f_chunk.read())
    # make sure both stacks are read
    assert full_actuals is not None
    assert full_raw_stack is not None
    assert chunk_actuals is not None
    assert chunk_raw_stack is not None

    # debug
    print(full_actuals)
    print(full_raw_stack)
    print(chunk_actuals)
    print(chunk_raw_stack)

    # read UW's profiling data is manycore_stats is defined
    with open(current_path + "/demo/manycore_stats.log", "r") as f_stats:
        external_stats = f_stats.read()
    assert external_stats is not None

    # read external cycles if necessary
    external_trim = None
    if external_stats is not None:
        data = external_stats.splitlines()
        overview = data[3]
        cycles = float(overview.split()[6])
        # convert to milliseconds
        ms = cycles / HB_FREQUENCY * 1000
        external_trim = ms

    # do the comparison
    aten_op = compare(full_raw_stack, chunk_raw_stack, external_trim=external_trim)
    cpu_log = aten_op.cpu_log
    hb_log = aten_op.hb_log

    # pytest assertion
    assert aten_op.hb_device_time == 1.204603
    assert aten_op.hb_host_time == 170.0
    assert aten_op.xeon_time == 34.451
    cpu_graph = stack_parser.exec_time_print_tree(cpu_log)
    assert cpu_graph == "|- Node(@CPU_LOG@ : 34.451)\n  |- Node(at::Tensor at::CPUType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>) : 0.043)\n  |- Node(at::native::add_stub::add_stub() : 34.316)"
    hb_graph = stack_parser.exec_time_print_tree(hb_log)
    assert hb_graph == "|- Node(@HB_LOG@ : 170.0)\n  |- Node(at::Tensor at::HammerBladeType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>) : 0.004)\n  |- Node(at::native::add_stub::add_stub() : 170.0)\n    |- Node(@OFFLOAD_KERNEL@__tensorlib_add : 0.0)\n      |- Node(@TRIM@ : 0.0)"
