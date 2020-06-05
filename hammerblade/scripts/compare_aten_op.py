"""
This script takes in 2 stacks and performs a comparison between CPU performance and HB performance
CPU performance is captured by processing the full input stack --full
HB performance is captured by processing the chunk input stack --chunk
External stats file can be passed in as --manycore-stats

06/04/2020 Lin Cheng (lc873@cornell.edu)
"""

import argparse
import copy
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
import std_parser
import stack_parser
import actual_parser
import process_CPU_stack
import process_HB_stack

# global variables

HB_FREQUENCY = 1000000000 # Hz = 1GHz

# INPUT:  NONE
# OUTPUT: parsed arguments

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default="full.std")
    parser.add_argument('--chunk', default="chunk.std")
    parser.add_argument('--manycore-stats', default=None)
    parser.add_argument('--fancy', action='store_true', default=False)
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

# INPUT:  aten op signature
# OUTPUT: fancy func name

def fancify(func):
    # if already in fancy format, return
    if func.find("aten") != -1:
        return func
    # if this is a special node, return the untouched func
    if func.startswith("@"):
        return func
    # fancify
    func = func.split("(")[0]
    func = func.split("::")[-1]
    func = "aten::" + func
    return func

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


class ATen_OP:
    def __init__(self, name, cpu_log, hb_log, time_on_hb, actuals):
        self.name = name
        self.cpu_log = copy.deepcopy(cpu_log)
        self.hb_log = copy.deepcopy(hb_log)
        # figure out time break down
        self.xeon_time = self.cpu_log.time
        self.hb_device_time = time_on_hb
        host_only_hb_log = copy.deepcopy(hb_log)
        def reset_trim(root):
            if root.func == "@TRIM@":
                root.time = float(0)
        host_only_hb_log = traversal(host_only_hb_log, reset_trim)
        stack_parser.exec_time_apply_trim(host_only_hb_log)
        self.hb_host_time = host_only_hb_log.time
        self.actuals = actuals

    def __str__(self):
        template = "\n|{func:^23}|{tensor:^15}|{full:^20}|{chunk:^20}|{xeon:^17}|{hb:^21}|{host:^17}|{device:^19}|"
        buf = """
+-----------------------+---------------+--------------------+--------------------+-----------------+---------------------+-----------------+-------------------+
|        ATen OP        |     Input     |     Full  Size     |     Chunk Size     |    Xeon Time    |    HB Total Time    |    Host Time    |    Device Time    |
+-----------------------+---------------+--------------------+--------------------+-----------------+---------------------+-----------------+-------------------+"""
        # this may not always true ... but ...
        assert len(self.actuals) > 0
        op_loc = 0
        loc = 0
        for actual in self.actuals:
            buf += template.format(
                func = fancify(self.name) if loc == op_loc else "",
                tensor = actual.name,
                full = actual.full,
                chunk = actual.chunk,
                xeon = "{:.2f}".format(self.xeon_time) if loc == op_loc else "",
                hb = "{:.2f}".format(self.hb_host_time + self.hb_device_time) if loc == op_loc else "",
                host = "{:.2f}".format(self.hb_host_time) if loc == op_loc else "",
                device = "{:.2f}".format(self.hb_device_time) if loc == op_loc else "")
            loc += 1
        buf += """
+-----------------------+---------------+--------------------+--------------------+-----------------+---------------------+-----------------+-------------------+
"""
        return buf


# INPUT:   full_raw_stack + full_actuals & chunk_raw_stack + chunk_actuals
# OUTPUT:  ATen_OP object
# OPTIONS: fancy_func -- show operator name as aten::op instead of raw func signature
#          external_trim -- use UW's profiler data instead of bsg_time for simulated time trimming

def compare_impl(full_raw_stack, full_actuals, chunk_raw_stack, chunk_actuals, fancy_func=False, external_trim=None):

    # cross check both stacks -- make sure they have the same shape
    cross_check(full_raw_stack, chunk_raw_stack)

    # build a full tree -- just to get the aten op name
    root = stack_parser.exec_time_tree(full_raw_stack, fancy_func=fancy_func)
    # this must be true -- the root has 2 children -- the aten op and other
    op_name = root.children[0].func

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

    # process input tensors
    actuals = actual_parser.parse(full_actuals, chunk_actuals)

    return ATen_OP(op_name, cpu_log, hb_log, total_time_on_HB, actuals)


# INPUT:   full out.std file path; chunk out.std file path; manycore_stats.log path
# OUTPUT:  ATen_OP object
# OPTIONS: fancy_func -- show operator name as aten::op instead of raw func signature

def compare(full_path, chunk_path, stats_path=None, fancy_func=False):

    # read actuals and stacks
    full_actuals = None
    full_raw_stack = None
    chunk_actuals = None
    chunk_raw_stack = None

    with open(full_path, "r") as f_full:
        full_actuals, full_raw_stack = std_parser.parse(f_full.read())
    with open(chunk_path, "r") as f_chunk:
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
    external_stats = None
    if stats_path is not None:
        with open(stats_path, "r") as f_stats:
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
    return compare_impl(full_raw_stack, full_actuals, chunk_raw_stack, chunk_actuals,
                        fancy_func=fancy_func, external_trim=external_trim)


# INPUT:   NONE
# OUTPUT:  NONE
# OPTIONS: --full -- full input size stack log
#          --chunk -- chuck input size stack log
#          --manycore-stats -- manycore_stats.log the log generated by UW's profiling tool

if __name__ == "__main__":

    args = parse_arguments()
    aten_op = compare(args.full, args.chunk, args.manycore_stats, args.fancy)
    print(aten_op)
