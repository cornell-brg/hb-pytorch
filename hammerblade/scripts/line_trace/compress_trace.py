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
#                                --startpc {optional} {Starting PC (of kernel) for the beginning of line trace}
#                                --endpc {optional} {Ending PC (of kernel) for the ending of line trace}
#                                --fastnfake {optional} {Flag to generate line trace of 4x4 Bladerunner}
#   Note: if startpc and endpc are not specified, line trace is generated for the entire trace file
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
    parser.add_argument("--startpc", type=str,
                        help="Starting PC of trace")
    parser.add_argument("--endpc", type=str, 
                        help="Ending PC of trace")
    parser.add_argument("--fastnfake", action="store_true", default=False,
                        help="Change machine config")
    
    args = parser.parse_args()

    if args.startpc is None:
        args.startpc = 'xx'
    if args.endpc is None:
        args.endpc = 'xx'
    
    # Creating a Trace object for all trace information between startpc and endpc
    TraceObj = Trace(args.trace, args.startpc, args.endpc, args.fastnfake)
    
    # Store the trace object in binary file
    trace_file = open('trace.obj', 'wb')
    pickle.dump(TraceObj, trace_file)

