# Copyright (c) 2019, University of Washington All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list
# of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# Neither the name of the copyright holder nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# TODO: Makefile comment
ORANGE=\033[0;33m
RED=\033[0;31m
NC=\033[0m

################################################################################
# Paths
################################################################################
_REPO_ROOT ?= $(shell git rev-parse --show-toplevel)
-include $(_REPO_ROOT)/environment.mk

################################################################################
# Include RISC-V Tool Configuration
################################################################################
-include $(FRAGMENTS_PATH)/kernel/tools.mk

_BSG_MANYCORE_SPMD_PATH = $(BSG_MANYCORE_DIR)/software/spmd/
_BSG_MANYCORE_CUDALITE_PATH = $(_BSG_MANYCORE_SPMD_PATH)/bsg_cuda_lite_runtime/
_BSG_MANYCORE_CUDALITE_MAIN_PATH = $(_BSG_MANYCORE_CUDALITE_PATH)/main

_BSG_MANYCORE_LIB_PATH    = $(BSG_MANYCORE_DIR)/software/bsg_manycore_lib
_BSG_MANYCORE_COMMON_PATH = $(_BSG_MANYCORE_SPMD_PATH)/common/

################################################################################
# C/C++ Compilation Flags
################################################################################
OPT_LEVEL ?= -O2
ARCH_OP   := rv32imaf
DEBUG_FLAGS := -g
# CCPPFLAGS are common between GCC and G++
RISCV_CCPPFLAGS += $(OPT_LEVEL)
RISCV_CCPPFLAGS += $(DEBUG_FLAGS)
RISCV_CCPPFLAGS += -march=$(ARCH_OP)
RISCV_CCPPFLAGS += -static
RISCV_CCPPFLAGS += -ffast-math
RISCV_CCPPFLAGS += -fno-common
RISCV_CCPPFLAGS += -ffp-contract=off

RISCV_CFLAGS   += -std=gnu99 $(RISCV_CCPPFLAGS)
RISCV_CXXFLAGS += -std=c++11 $(RISCV_CCPPFLAGS)

RISCV_INCLUDES += -I$(_BSG_MANYCORE_COMMON_PATH)
RISCV_INCLUDES += -I$(BSG_MANYCORE_DIR)/software/bsg_manycore_lib

RISCV_DEFINES += -Dbsg_global_X=$(BSG_MACHINE_GLOBAL_X)
RISCV_DEFINES += -Dbsg_global_Y=$(BSG_MACHINE_GLOBAL_Y)
RISCV_DEFINES += -Dbsg_group_size=$(_BSG_MACHINE_TILES)
RISCV_DEFINES += -DPREALLOCATE=0
RISCV_DEFINES += -DHOST_DEBUG=0

# We build and name a machine-specific crt.rvo because it's REALLY
# difficult to figure out why your program/cosimulation is hanging
# when the wrong link script was used during linking
MACHINE_CRT_OBJ = $(BSG_MACHINE_NAME).rvo
$(MACHINE_CRT_OBJ) crt.rvo: $(_BSG_MANYCORE_COMMON_PATH)/crt.S $(BSG_MACHINE_PATH)/Makefile.machine.include
	$(RISCV_GCC) $(RISCV_CFLAGS) $(RISCV_DEFINES) $(RISCV_INCLUDES) -c $< -o $@ |& tee $*.comp.log

# We compile these locally so that we don't interfere with the files in
# $(_BSG_MANYCORE_LIB_PATH). They are not architecture specific, and not
# tile-group-size specific, but we do have to define them... soo...

# The following two lines that define bsg_tiles_X and bsg_tiles_Y for
# bsg_printf.rvo *** IS A HACK ***. They aren't used in any source file or
# function that CUDA uses, but bsg_manycore.h will FAIL to compile if they
# aren't defined because they are used in macros.
bsg_set_tile_x_y.rvo bsg_printf.rvo main.rvo: RISCV_DEFINES += -Dbsg_tiles_X=$(_BSG_MACHINE_TILES_X)
bsg_set_tile_x_y.rvo bsg_printf.rvo main.rvo: RISCV_DEFINES += -Dbsg_tiles_Y=$(_BSG_MACHINE_TILES_Y)

bsg_set_tile_x_y.rvo bsg_printf.rvo: %.rvo:$(_BSG_MANYCORE_LIB_PATH)/%.c
	$(RISCV_GCC) $(RISCV_CFLAGS) $(RISCV_DEFINES) $(RISCV_INCLUDES) -c $< -o $@

main.rvo: $(_BSG_MANYCORE_CUDALITE_MAIN_PATH)/main.c
	$(RISCV_GCC) $(RISCV_CFLAGS) $(RISCV_DEFINES) $(RISCV_INCLUDES) -c $< -o $@

ifeq ($(_KERNEL_COMPILER), GCC)
  -include $(FRAGMENTS_PATH)/kernel/gcc/compile.mk
else ifeq ($(_KERNEL_COMPILER), CLANG)
  -include $(FRAGMENTS_PATH)/kernel/clang/compile.mk
else ifndef _KERNEL_COMPILER
  $(error $(shell echo -e "$(RED)BSG MAKE ERROR: Variable _KERNEL_COMPILER undefined. Must be GCC or CLANG$(NC)")
else
  $(error $(shell echo -e "$(RED)BSG MAKE ERROR: Invalid value for variable _KERNEL_COMPILER. Was $(_KERNEL_COMPILER). Must be GCC or CLANG$(NC)")
endif

.PRECIOUS: %.rvo
