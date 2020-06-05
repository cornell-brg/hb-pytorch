"""
This script takes in 2 stacks and performs a comparison between CPU performance and HB performance
CPU performance is captured by processing the full input stack --full
HB performance is captured by processing the chunk input stack --chunk
External stats file can be passed in as --manycore-stats

06/04/2020 Lin Cheng (lc873@cornell.edu)
"""

import argparse
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
import stack_parser
import process_CPU_stack
import process_HB_stack

# global variables

HB_FREQUENCY = 1000000000 # Hz = 1GHz

# INPUT:  NONE
# OUTPUT: parsed arguments

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default="full.stack")
    parser.add_argument('--chunk', default="chunk.stack")
    parser.add_argument('--manycore-stats', default="NONE")
    return parser.parse_args()


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

def cross_check(full_raw_stack, chunk_raw_stack):
    full_tree = stack_parser.exec_time_tree(full_raw_stack)
    chunk_tree = stack_parser.exec_time_tree(chunk_raw_stack)
    full_func_list = []
    def gather_full_func(root):
        full_func_list.append(root.func)
    chunk_func_list = []
    def gather_chunk_func(root):
        chunk_func_list.append(root.func)
    traversal(full_tree, gather_full_func)
    traversal(chunk_tree, gather_chunk_func)
    assert len(full_func_list) == len(chunk_func_list)
    full_func = "<|>".join(full_func_list)
    chunk_func = "<|>".join(chunk_func_list)
    assert full_func == chunk_func


# INPUT:   full_raw_stack & chunk_raw_stack
# OUTPUT:  (CPU_tree, HB_tree, time on HB)
# OPTIONS: fancy_func -- show operator name as aten::op instead of raw func signature
#          external_trim -- use UW's profiler data instead of bsg_time for simulated time trimming

def compare(full_raw_stack, chunk_raw_stack, fancy_func=False, external_trim=None):

    # cross check both stacks -- make sure they have the same shape
    cross_check(full_raw_stack, chunk_raw_stack)

    # process CPU_log and HB_log
    # CPU_log should be given by full input data
    # HB_log should be given by chunk input data
    cpu_log = process_CPU_stack.parse(full_raw_stack, fancy_func=fancy_func)
    hb_log = process_HB_stack.parse(chunk_raw_stack, fancy_func=fancy_func, trimming=True)

    # re-apply trim of 0 if using external tirmming is enabled
    # since we can have more than one @TRIM@ node, we cannot just adjust these nodes
    if external_trim is not None:
        def reset_trim(root):
            if root.func == "@TRIM@":
                root.time = float(0)
        hb_log = traversal(hb_log, reset_trim)
        stack_parser.exec_time_apply_trim(hb_log)

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
    print(stack_parser.exec_time_print_tree(cpu_log))
    print(stack_parser.exec_time_print_tree(hb_log))
    print("total time on HB = " + str(total_time_on_HB))

    return (cpu_log, hb_log, total_time_on_HB)

# INPUT:   NONE
# OUTPUT:  NONE
# OPTIONS: --full -- full input size stack log
#          --chunk -- chuck input size stack log
#          --manycore-stats -- manycore_stats.log the log generated by UW's profiling tool

if __name__ == "__main__":

    args = parse_arguments()

    # read stacks
    full_raw_stack = None
    chunk_raw_stack = None
    with open(args.full, "r") as f_full:
        full_raw_stack = f_full.read()
    with open(args.chunk, "r") as f_chunk:
        chunk_raw_stack = f_chunk.read()
    # make sure both stacks are read
    assert full_raw_stack is not None
    assert chunk_raw_stack is not None

    # read UW's profiling data is manycore_stats is defined
    external_stats = None
    if args.manycore_stats != "NONE":
        with open(args.manycore_stats, "r") as f_stats:
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
    compare(full_raw_stack, chunk_raw_stack, external_trim=external_trim)
