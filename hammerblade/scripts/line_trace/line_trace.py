#
#   line_trace.py
#
#   Common header file for line trace generator
#   Note: Do not use directly; use compress_trace.py and print_trace.py as described below
#
#   @author: Krithik Ranjan (kr397@cornell.edu) 08/19/20
#
#   Usage: 
#
#   Step 1: Compress the .csv trace file with compress_trace.py
#       python compress_trace.py --trace {vanilla_operaton_trace.csv}
#                                --startpc {Starting PC (of kernel) for the beginning of line trace}
#                                --endpc {Ending PC (of kernel) for the ending of line trace}
#                                --fastnfake {optional} {Flag to generate line trace of 4x4 Bladerunner}
#   Example:
#       python compress_trace.py --trace vanilla_operation_trace.py
#                                --startpc 1cd2c
#                                --endpc 1cd80
#
#   Step 2: Prints the line trace between two PCs specified while running compress_trace.py
#   
#   For every tile, the instructions can be printed in either 'lo', 'mid', 'hi' 
#   - lo: single character code for stalls (listed below), '@' for instruction
#   - mid: three character code for stalls and instructions (listed below)
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
#   Note: compress_trace.py generates a file trace.obj in the active directory; ensure that
#   print_trace.py is run from the same directory so that it can access trace.obj
#


import csv
import argparse

class Trace:

    _TILE_X_DIM = 16
    _TILE_Y_DIM = 8

    # List of types of stalls incurred by the core
    # Stall type : [lo code, mid code]
    _STALLS_LIST   = {"stall_depend_dram_load" : ['l', '#dl'],
                      "stall_depend_group_load" : ['l', '#gl'] ,
                      "stall_depend_global_load" : ['l', '#gl'],
                      "stall_depend_local_load" : ['l', '#ll'],

                      "stall_depend_idiv" : ['d', '#id'],
                      "stall_depend_fdiv" : ['d', '#fd'],
                      "stall_depend_imul" : ['d', '#im'],

                      "stall_amo_aq" : ['a', '#aq'],
                      "stall_amo_rl" : ['a', '#rl'],

                      "stall_bypass" : ['r', '#bp'],
                      "stall_lr_aq" : ['r', '#lr'],
                      "stall_fence" : ['r', '#fe'],
                      "stall_remote_req" : ['r', '#rr'],
                      "stall_remote_credit" : ['r', '#rc'],

                      "stall_fdiv_busy" : ['b', '#fb'],
                      "stall_idiv_busy" : ['b', '#ib'],

                      "stall_fcsr" : ['f', '#fc'],
                      "stall_remote_ld" : ['f', '#ld'],

                      "stall_remote_flw_wb" : ['w', '#fl'],

                      "bubble_branch_miss" : ['j', '#bm'],
                      "bubble_jalr_miss" : ['j', '#jm'],

                      "stall_ifetch_wait" : ['i', '#if'],
                      "bubble_icache_miss" : ['i', '#ic'],
                      "icache_miss" : ['i', '#im']}


    # List of types of integer instructions executed by the core
    _INSTRS_LIST    = ["local_ld" : 'lld',
                       "local_st": 'lst',
                       "remote_ld_dram": 'ldr',
                       "remote_ld_global": 'lgl',
                       "remote_ld_group": 'lgr',
                       "remote_st_dram": 'sdr',
                       "remote_st_global": 'sgl',
                       "remote_st_group": 'sgr',
                       "local_flw": 'flw',
                       "local_fsw": 'fsw',
                       "remote_flw_dram": 'fdr',
                       "remote_flw_global": 'fgl',
                       "remote_flw_group": 'fgr',
                       "remote_fsw_dram": 'fsd',
                       "remote_fsw_global": 'fsg',
                       "remote_fsw_group": 'fsr',
                       # icache_miss is no longer treated as an instruction
                       # but treated the same as stall_ifetch_wait
                       # "icache_miss",
                       "lr": 'lr ',
                       "lr_aq": 'lra',
                       "amoswap": 'ams',
                       "amoor": 'amo',
                       "beq": 'beq',
                       "bne": 'bne',
                       "blt": 'blt',
                       "bge": 'bge',
                       "bltu": 'blu',
                       "bgeu": 'bgu',
                       "jal": 'jal',
                       "jalr": 'jar',
                       "beq_miss": 'eqm',
                       "bne_miss": 'nem',
                       "blt_miss": 'ltm',
                       "bge_miss": 'gem',
                       "bltu_miss": 'lum',
                       "bgeu_miss": 'gum',
                       "jalr_miss": 'jam',
                       "sll": 'sll',
                       "slli": 'sli',
                       "srl": 'srl',
                       "srli": 'sri',
                       "sra": 'sra',
                       "srai": 'sai',
                       "add": 'add',
                       "addi": 'adi',
                       "sub": 'sub',
                       "lui": 'lui',
                       "auipc": 'apc',
                       "xor": 'xor',
                       "xori": 'xri',
                       "or": 'or ',
                       "ori": 'ori',
                       "and": 'and',
                       "andi": 'ani',
                       "slt": 'slt',
                       "slti": 'sli',
                       "sltu": 'slu',
                       "sltiu": 'siu',
                       "div": 'div',
                       "divu": 'diu',
                       "rem": 'rem',
                       "remu": 'reu',
                       "mul": 'mul',
                       "fence": 'fen',
                       "csrrw": 'crw',
                       "csrrs": 'crs',
                       "csrrc": 'crc',
                       "csrrwi": 'cwi',
                       "csrrsi": 'csi',
                       "csrrci": 'cci',
                       "unknown": 'unk']


    # List of types of floating point instructions executed by the core
    _FP_INSTRS_LIST = ["fadd": 'fad',
                       "fsub": 'fsu',
                       "fmul": 'fmu',
                       "fsgnj": 'fgj',
                       "fsgnjn": 'fjn',
                       "fsgnjx": 'fjx',
                       "fmin": 'fmi',
                       "fmax": 'fma',
                       "fcvt_s_w": 'fcw',
                       "fcvt_s_wu": 'fcu',
                       "fmv_w_x": 'fwx',
                       "fmadd": 'fma',
                       "fmsub": 'fms',
                       "fnmsub": 'fns',
                       "fnmadd": 'fna',
                       "feq": 'feq',
                       "flt": 'flt',
                       "fle": 'fle',
                       "fcvt_w_s": 'fcs',
                       "fcvt_wu_s": 'fus',
                       "fclass": 'fcl',
                       "fmv_x_w": 'fxw',
                       "fdiv": 'fdi',
                       "fsqrt": 'fsq']

    def __init__(self, trace_file, tile_list, start_pc, end_pc, ff):
        if ff:
            self._TILE_Y_DIM = 4
            self._TILE_X_DIM = 4
        self.start_cycle = -1
        self.end_cycle = -1

        # Parse vanilla trace file to generate traces
        self.traces = self.__parse_traces(trace_file, start_pc, end_pc)


    
    def __parse_traces(self, trace_file, start_pc, end_pc):
        traces = {}
        
        for tile in range(self._TILE_X_DIM * self._TILE_Y_DIM):
            tile_x = tile % self._TILE_X_DIM
            tile_y = tile // self._TILE_X_DIM
            traces[(tile_x, tile_y)] = {"inrange": False, "mode": 'na', "instr": {}}

        with open(trace_file) as csv_trace:
            reader = csv.DictReader(csv_trace, delimiter=',')

            for row in reader:
                tile_x = int(row["x"])
                tile_y = int(row["y"])

                if (tile_x, tile_y) in traces:
                    if row["pc"] == start_pc:
                        traces[(tile_x, tile_y)]["inrange"] = True
                        if self.start_cycle == -1:
                            self.start_cycle = int(row["cycle"])
                    
                    if traces[(tile_x, tile_y)]["inrange"]:
                        traces[(tile_x, tile_y)]["instr"][int(row["cycle"])] = row["operation"]
                    
                    if row["pc"] == end_pc:
                        self.end_cycle = int(row["cycle"])
                        traces[(tile_x, tile_y)]["inrange"] = False

        return traces            

    def set_mode(self, mode, start, end, all_tiles = False):
        if all_tiles:
            start = 0
            end = self._TILE_X_DIM * self._TILE_Y_DIM

        for tile in range(start, end):
            tile_x = tile % self._TILE_X_DIM
            tile_y = tile // self._TILE_X_DIM
            self.traces[(tile_x, tile_y)]["mode"] = mode

    def print_trace(self):
        print("Start cycle: " + str(self.start_cycle) + " End cycle: " + str(self.end_cycle))

        print("Tiles", end='\t')
        for tile in self.traces:
            if self.traces[tile]["mode"]  == 'lo':
                if tile[0] >= 10:
                    print(hex(tile[0]).lstrip('0x'), end='')
                else:
                    print(tile[0], end='')
            elif self.traces[tile]["mode"]  == 'mid':
                if tile[0] >= 10:
                    print(hex(tile[0]).lstrip('0x').ljust(3), end=' ')
                else:
                    print(str(tile[0]).ljust(3), end=' ')
            elif self.traces[tile]["mode"]  == 'hi':
                if tile[0] >= 10:
                    print(hex(tile[0]).lstrip('0x').ljust(15), end=' ')
                else:
                    print(str(tile[0]).ljust(15), end=' ')

        print("\nCycles", end='\t')
        for tile in self.traces:
            if self.traces[tile]["mode"] == 'lo':
                print(str(tile[1]), end='')
            elif self.traces[tile]["mode"] == 'mid':
                print(str(tile[1]).ljust(3), end=' ')
            elif self.traces[tile]["mode"] == 'hi':
                print(str(tile[1]).ljust(15), end=' ')
            
        print()

        for cycle in range(self.start_cycle, self.end_cycle+1):
            print(cycle, end='\t')

            for tile in self.traces:
                self.__print_op(self.traces[tile], cycle)
            print()
        
                
    def __print_op(self, tile_trace, cycle):
        if tile_trace["mode"] == 'lo':
            if cycle in tile_trace["instr"]:
                op = tile_trace["instr"][cycle]
                if op in self._INSTRS_LIST or op in self._FP_INSTRS_LIST:
                    print('@', end='')
                elif op in self._STALLS_LIST:
                    print(self._STALLS_LIST[op][0], end='')
                else:
                    print('0', end='')
            else :
                print(' ', end='')
        elif tile_trace["mode"] == 'mid':
            if cycle in tile_trace["instr"]:
                op = tile_trace["instr"][cycle]
                if op in self._INSTRS_LIST:
                    print(self._INSTRS_LIST[op], end=' ')
                elif op in self._FP_INSTRS_LIST:
                    print(self._FP_INSTRS_LIST[op], end=' ')
                elif op in self._STALLS_LIST:
                    print(self._STALLS_LIST[op][1], end=' ')
                else:
                    print('000', end=' ')
            else:
                print('   ', end=' ')
        elif tile_trace["mode"] == 'hi':
            if cycle in tile_trace["instr"]:
                op_len = 15
                op = tile_trace["instr"][cycle]
                if len(op) < op_len:
                    print(op.ljust(op_len), end=' ')
                else:
                    print(op[:op_len], end=' ')
            else:
                print('               ', end=' ')
            
