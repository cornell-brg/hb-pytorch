#
#   compress_asm.py
#
#   Compress the disassembly of PyTorch
#   Optional Step 1b in generation of line trace (only required for 'full' mode)
#
#   @author: Krithik Ranjan (kr397@cornell.edu) 08/31/20
#
#   Usage:
#       python compress_asm.py --asm {Path to disassembly}
#   Example: 
#       python compress_asm.py --asm /work/global/kr397/hb-pytorch/torch/riscv/kernel.dis 
#
#   Note: Refer to line_trace.py for complete instructions to generate line traces
#   Must be run before running print_trace.py in 'full' mode; in the same directory
#

import pickle
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--asm", type=str,
                        help="Path to disassembly file")
    args = parser.parse_args()

    # Disassembly file
    disasm = open(args.asm, 'r')

    dic = {}
    for line in disasm:
        instr = line.strip().split('\t')

        # Differentiate instruction lines from other lines
        if len(instr) > 2 :
            pc = instr[0][:-1]
            # Add the instruction to dictionary
            if pc not in dic:
                dic[pc] = instr[2:]

    asmdic = open('kernel.dic', 'wb')
    pickle.dump(dic, asmdic)
    asmdic.close()
