"""
This script takes in a stack and looks for the CPU_log
We use this script to process the full input performance on CPU
per ATen Operator

06/04/2020 Lin Cheng (lc873@cornell.edu)
"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
import stack_parser

# INPUT:  raw stack of a single ATen Operator
# OUTPUT: subtree with CPU_log as root
# OPTIONS: fancy_func -- show operator name as aten::op instead of raw func signature

def parse(raw_stack, fancy_func=False):
    root = stack_parser.exec_time_tree(raw_stack, fancy_func=fancy_func)
    # in the per ATen OP context, the tree looks like
    #            root
    #        /          \
    #   aten::op      other
    assert len(root.children) == 2
    assert root.children[1].func == "other"
    aten_op = root.children[0]
    # look for @CPU_LOG@
    # there has to be a CPU_log to use this parser
    # the aten_op tree should look like
    #            aten::op
    #      /        |        \
    #   CPU_log  ******   HB_log
    cpu_log = None
    for child in aten_op.children:
      if child.func == "@CPU_LOG@":
          cpu_log = child
          break
    assert cpu_log is not None
    return cpu_log
