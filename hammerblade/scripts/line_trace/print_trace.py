#
#   print_trace.py
#
#   Prints the line trace between two PCs specified while running compress_trace.py
#   Step 2 in line trace generation
#
#   @author: Krithik Ranjan (kr397@cornell.edu) 08/19/20
#   
#   For every tile, the instructions can be printed in either 'lo', 'mid', 'hi' 
#   - lo: single character code for stalls (listed in line_trace.py), '@' for instruction
#   - mid: three character code for stalls and instructions (listed in line_trace.py) 
#   - hi: first fifteen characters of all stalls and instructions
#       
#       python print_trace.py --mode {optional} {print mode for all tiles; 'lo', 'mid', 'hi'}
#                             --lo {optional} {range of tiles to print in 'lo' mode}
#                             --mid {optional} {range of tiles to print in 'mid' mode}
#                             --hi {optional} {range of tiles to print in 'hi' mode}
#   
#   Note: at least one of mode, lo, mid, hi must be provided to print the line trace
#
#   Example:
#       python print_trace.py --mode lo
#           (prints the line trace of all tiles in lo mode)
#       python print_trace.py --mode lo
#                             --mid 4 9
#           (prints the line trace of tiles [4-9) in mid, and all else in lo)
#       python print_trace.py --lo 0 3
#                             --mid 5 9
#                             --hi 10 11
#           (prints the line trace of tiles [0-3) in lo, [5-9) in mid, [10-11) in hi)
#
#   Note: Must be run after compress_trace.py, in the same directory
#   Refer to line_trace.py for complete line trace generation instructions
#


from line_trace import Trace
import pickle
import argparse
import csv


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, 
                        help="Mode to print the trace of all tiles")
    parser.add_argument("--lo", nargs=2, type=int,  
                        help="Range of tiles to print in lo-res")
    parser.add_argument("--mid", nargs=2, type=int,  
                        help="Range of tiles to print in mid-res")
    parser.add_argument("--hi", nargs=2, type=int,  
                        help="Range of tiles to print in hi-res")


    # Read trace object from trace.obj
    trace_file = open('trace.obj', 'rb')
    TraceObj = pickle.load(trace_file)

    # Parse the range arguments and set up mode
    args = parser.parse_args()
    if args.mode is not None:
        TraceObj.set_mode(args.mode, 0, 0, all_tiles=True)
    if args.lo is not None:
        TraceObj.set_mode('lo', args.lo[0], args.lo[1])
    if args.mid is not None:
        TraceObj.set_mode('mid', args.mid[0], args.mid[1])
    if args.hi is not None:
        TraceObj.set_mode('hi', args.hi[0], args.hi[1])

    if args.mode is None and args.lo is None and args.mid is None and args.hi is None:
        print("Incorrect arguments")
    else:
        TraceObj.print_trace()

