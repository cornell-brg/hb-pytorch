"""
This script takes in a stack and looks for the HB_log
We use this script to process the chunk input performance on HammerBlade
per ATen Operator

06/04/2020 Lin Cheng (lc873@cornell.edu)
"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
import stack_parser

# INPUT:   raw stack of a single ATen Operator
# OUTPUT:  subtree with HB_log as root
# OPTIONS: fancy_func -- show operator name as aten::op instead of raw func signature
#          trimming -- apply trimming

def parse(raw_stack, fancy_func=False, trimming=False):
    root = stack_parser.exec_time_tree(raw_stack, fancy_func=fancy_func, trimming=trimming)
    # in the per ATen OP context, the tree looks like
    #            root
    #        /          \
    #   aten::op      other
    assert len(root.children) == 2
    assert root.children[1].func == "other"
    aten_op = root.children[0]
    # look for @HB_LOG@
    # there has to be a HB_log to use this parser
    # the aten_op tree should look like
    #            aten::op
    #      /        |        \
    #   CPU_log  ******   HB_log
    hb_log = None
    for child in aten_op.children:
      if child.func == "@HB_LOG@":
          hb_log = child
          break
    assert hb_log is not None
    return hb_log
