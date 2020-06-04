"""
This script takes in 2 stacks and performs a comparison between CPU performance and HB performance
CPU performance is captured by processing the full input stack --full
HB performance is captured by processing the chunk input stack --chunk

06/04/2020 Lin Cheng (lc873@cornell.edu)
"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
import stack_parser
import process_CPU_stack
import process_HB_stack

# INPUT:   a stack tree, func f that should be applied to each node
# OUTPUT:  modified stack tree
# OPTIONS: NONE
# NOTE:    pre-order traversal

def traversal(root, func):
    func(root)
    for child in root.children:
        traversal(child, func)
    return root

# INPUT:   full_raw_stack & chunk_raw_stack
# OUTPUT:  NONE
# OPTIONS: fancy_func -- show operator name as aten::op instead of raw func signature
#          external_trim -- use UW's profiler data instead of bsg_time for simulated time trimming

def compare(full_raw_stack, chunk_raw_stack, fancy_func=False, external_trim=None):
    # process CPU_log and HB_log
    # CPU_log should be given by full input data
    # HB_log should be given by chunk input data
    cpu_log = process_CPU_stack.parse(full_raw_stack, fancy_func=fancy_func)
    hb_log = process_HB_stack.parse(full_raw_stack, fancy_func=fancy_func, trimming=True)

    # debug
    print(stack_parser.exec_time_print_tree(cpu_log))
    print(stack_parser.exec_time_print_tree(hb_log))

    # re-apply trim of 0 if using external tirmming is enabled
    # since we can have more than one @TRIM@ node, we cannot just adjust these nodes
    if external_trim is not None:
        def reset_trim(root):
            if root.func == "@TRIM@":
                root.time = float(0)
        hb_log = traversal(hb_log, reset_trim)
        stack_parser.exec_time_apply_trim(hb_log)

    # debug
    print(stack_parser.exec_time_print_tree(hb_log))

    # get total time on device
    # so we accumulate all simualted time
    total_time_on_HB = 0
    if external_trim is None:
        def acc_trim(root):
            nonlocal total_time_on_HB
            if root.func == "@TRIM@":
                total_time_on_HB += root.time
        traversal(hb_log, acc_trim)
    else:
        total_time_on_HB = external_trim

    # debug
    print("total time on HB = " + str(total_time_on_HB))






# ad-hoc testing
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
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>@HB_LOG@<|>at::native::add_stub::add_stub()<|>@OFFLOAD_KERNEL@__tensorlib_add<|>@TRIM@;42
at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)<|>at::Tensor at::CPUType::{anonymous}::llcopy(const at::Tensor&);0.025

#TOP_LEVEL_FUNC_END#__at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)
"""

compare(raw_stack, raw_stack, external_trim=0.3154)
print("==================")
compare(raw_stack, raw_stack)

