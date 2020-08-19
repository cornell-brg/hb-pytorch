#
#   compress_trace.py
#
#   Compress the .csv trace file
#   Step 1 in generation of line trace
#
#   @author: Krithik Ranjan (kr397@cornell.edu) 08/19/20
#
#   Usage:
#       python compress_trace.py --trace {vanilla_operaton_trace.csv}
#                                --startpc {Starting PC (of kernel) for the beginning of line trace}
#                                --endpc {Ending PC (of kernel) for the ending of line trace}
#                                --fastnfake {optional} {Flag to generate line trace of 4x4 Bladerunner}
#   Example:
#       python compress_trace.py --trace vanilla_operation_trace.py
#                                --startpc 1cd2c
#                                --endpc 1cd80
#
#   Note: Refer to line_trace.py for complete instructions to generate line traces
#   Must be run before running print_trace.py
#

from line_trace import Trace
import pickle
import argparse
import csv


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--trace", default="vanilla_operation_trace.csv", type=str,
                        help="Vanilla operation log file")
    parser.add_argument("--startpc", type=str, required=True, 
                        help="Starting PC of trace")
    parser.add_argument("--endpc", type=str, required=True, 
                        help="Ending PC of trace")
    parser.add_argument("--fastnfake", action="store_true", default=False,
                        help="Change machine config")
    
    args = parser.parse_args()
    tile_list = list(range(128))
    
    # Creating a Trace object for all trace information between startpc and endpc
    TraceObj = Trace(args.trace, tile_list, args.startpc, args.endpc, args.fastnfake)
    
    # Store the trace object in binary file
    trace_file = open('trace.obj', 'wb')
    pickle.dump(TraceObj, trace_file)

