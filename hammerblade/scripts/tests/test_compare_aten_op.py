import sys
import pathlib
current_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(current_path + '/..')
import stack_parser
from compare_aten_op import HB_FREQUENCY, compare

def test_compare_aten_op_1():
    # read stacks
    full_raw_stack = None
    chunk_raw_stack = None
    with open(current_path + "/demo/full.stack", "r") as f_full:
        full_raw_stack = f_full.read()
    with open(current_path + "/demo/chunk.stack", "r") as f_chunk:
        chunk_raw_stack = f_chunk.read()
    # make sure both stacks are read
    assert full_raw_stack is not None
    assert chunk_raw_stack is not None

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
    cpu_log, hb_log, time_on_hb = compare(full_raw_stack, chunk_raw_stack, external_trim=external_trim)

    # pytest assertion
    assert time_on_hb == 1.204603
    cpu_graph = stack_parser.exec_time_print_tree(cpu_log)
    assert cpu_graph == "|- Node(@CPU_LOG@ : 34.895)\n  |- Node(at::Tensor at::CPUType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>) : 0.031)\n  |- Node(at::native::add_stub::add_stub() : 34.707)"
    hb_graph = stack_parser.exec_time_print_tree(hb_log)
    assert hb_graph == "|- Node(@HB_LOG@ : 170.0)\n  |- Node(at::Tensor at::HammerBladeType::{anonymous}::empty(c10::IntArrayRef, const c10::TensorOptions&, c10::optional<c10::MemoryFormat>) : 0.004)\n  |- Node(at::native::add_stub::add_stub() : 170.0)\n    |- Node(@OFFLOAD_KERNEL@__tensorlib_add : 0.0)\n      |- Node(@TRIM@ : 0.0)"
