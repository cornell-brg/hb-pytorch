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

_LINK_HELP_STRING := "Rules included from kernel/link.mk\n"

################################################################################
# Paths / Environment Configuration
################################################################################
_REPO_ROOT ?= $(shell git rev-parse --show-toplevel)
-include $(_REPO_ROOT)/hammerblade/environment.mk

################################################################################
# BSG Manycore Make Functions
################################################################################
-include $(FRAGMENTS_PATH)/functions.mk

################################################################################
# BSG Manycore Machine Configuration
################################################################################
# Import configuration from machine.mk. Defines Architectural
# parameters: Architecture Dimensions (BSG_MACHINE_GLOBAL_Y,
# BSG_MACHINE_GLOBAL_X), DRAM Present (BSG_DRAM_INCLUDED), DRAM Bank
# Size (BSG_MACHINE_DRAM_BANK_SIZE_WORDS), Victim Cache Block Size
# (BSG_MACHINE_VCACHE_BLOCK_SIZE_WORDS), Victim Cache Ways
# (BSG_MACHINE_VCACHE_WAY), Victim Cache Number of Sets
# (BSG_MACHINE_VCACHE_SET).
-include $(FRAGMENTS_PATH)/machine.mk

################################################################################
# BSG Manycore Tools Configuration
################################################################################
# Import tools configuration from tools.mk. This sets RISCV_GCC,
# RISCV_GXX, RISCV_LINK and other RISCV toolchain variables
-include $(FRAGMENTS_PATH)/kernel/tools.mk

################################################################################
# ELF File Parameters
################################################################################
# Default .data section location; LOCAL=>DMEM, SHARED=>DRAM.
BSG_ELF_DEFAULT_DATA_LOC ?= LOCAL

BSG_ELF_OFF_CHIP_MEM := $(BSG_MACHINE_DRAM_INCLUDED)

# Total addressable DRAM size (in 32-bit WORDS, and SIZE bytes)
BSG_ELF_DRAM_WORDS := $(shell expr $(BSG_MACHINE_DRAM_BANK_SIZE_WORDS) \* $(BSG_MACHINE_GLOBAL_X))
BSG_ELF_DRAM_SIZE := $(shell expr $(BSG_ELF_DRAM_WORDS) \* 4)

# Victim Cache Set Size (in 32-bit WORDS and SIZE bytes)
_BSG_ELF_VCACHE_SET_WORDS := $(shell expr $(BSG_MACHINE_VCACHE_WAY) \* $(BSG_MACHINE_VCACHE_BLOCK_SIZE_WORDS))
BSG_ELF_VCACHE_SET_SIZE := $(shell expr $(_BSG_ELF_VCACHE_SET_WORDS) \* 4)

# Victim Cache Column Size (in 32-bit WORDS and SIZE bytes)
_BSG_ELF_VCACHE_COLUMN_WORDS := $(shell expr $(BSG_MACHINE_VCACHE_SET) \* $(_BSG_ELF_VCACHE_SET_WORDS))
BSG_ELF_VCACHE_COLUMN_SIZE := $(shell expr $(_BSG_ELF_VCACHE_COLUMN_WORDS) \* 4)

# Victim Cache Total Size (in 32-bit WORDS, and SIZE BYTES)
_BSG_ELF_VCACHE_MANYCORE_WORDS ?= $(shell expr $(BSG_MACHINE_GLOBAL_X) \* $(_BSG_ELF_VCACHE_COLUMN_WORDS))
BSG_ELF_VCACHE_MANYCORE_SIZE := $(shell expr $(_BSG_ELF_VCACHE_MANYCORE_WORDS) \* 4)

# EVA of stack pointer
BSG_ELF_DRAM_EVA_OFFSET = 0x80000000

# Compute the ELF Stack Pointer Location. If the .data segment is in
# DMEM (LOCAL) then put it at the top of DMEM. Otherwise, use the top
# of DRAM (if present), or the Victim Cache address space (if DRAM is
# disabled/not present).
ifeq ($(BSG_ELF_DEFAULT_DATA_LOC), LOCAL)
BSG_ELF_STACK_PTR ?= 0x00001ffc
else
  ifeq ($(BSG_ELF_OFF_CHIP_MEM), 1)
  _BSG_ELF_DRAM_LIMIT = $(shell expr $(BSG_ELF_DRAM_EVA_OFFSET) + $(BSG_ELF_DRAM_SIZE))
  else
  _BSG_ELF_DRAM_LIMIT = $(shell expr $(BSG_ELF_DRAM_EVA_OFFSET) + $(BSG_ELF_VCACHE_MANYCORE_SIZE))
  endif
# Stack Pointer Address: Subtract 4 from the maximum memory space
# address address
BSG_ELF_STACK_PTR = $(shell expr $(_BSG_ELF_DRAM_LIMIT) - 4)
endif

# Determine the correct linker script from machine parameters. The
# presence of DRAM/Off-Chip memory, and the location of the .data
# segment determine which (poorly named) link script will be
# chosen. Errors are thrown if an incorrect configuration is found
RISCV_LINK_SCRIPT=foo
ifeq ($(BSG_ELF_OFF_CHIP_MEM), 1)
  ifeq ($(BSG_ELF_DEFAULT_DATA_LOC), LOCAL)
    RISCV_LINK_SCRIPT := $(BSG_MANYCORE_DIR)/software/spmd/common/link_dmem.ld
  else ifeq ($(BSG_ELF_DEFAULT_DATA_LOC), SHARED)
    RISCV_LINK_SCRIPT := $(BSG_MANYCORE_DIR)/software/spmd/common/link_dram.ld
  else
    $(error $(shell echo -e "$(RED)Invalid BSG_ELF_DEFAULT_DATA_LOC = $(BSG_ELF_DEFAULT_DATA_LOC); Only LOCAL and SHARED are valid $(NC)"))
  endif
else ifeq ($(BSG_ELF_OFF_CHIP_MEM), 0)
  ifeq ($(BSG_ELF_DEFAULT_DATA_LOC), LOCAL)
    RISCV_LINK_SCRIPT := $(BSG_MANYCORE_DIR)/software/spmd/common/link_dmem2.ld
  else ifeq ($(BSG_ELF_DEFAULT_DATA_LOC), SHARED)
    RISCV_LINK_SCRIPT := $(BSG_MANYCORE_DIR)/software/spmd/common/link_dram2.ld
  else
    $(error $(shell echo -e "$(RED)Invalid BSG_ELF_DEFAULT_DATA_LOC = $(BSG_ELF_DEFAULT_DATA_LOC); Only LOCAL and SHARED are valid$(NC)"))
  endif
else
  $(error $(shell echo -e "$(RED)Invalid BSG_ELF_OFF_CHIP_MEM = $(BSG_ELF_OFF_CHIP_MEM); Only 0 and 1 are valid$(NC)"))
endif

# BSG Manycore Library Variables
BSG_MANYCORE_LIB_OBJECTS  += bsg_set_tile_x_y.rvo
# We don't include bsg_printf.rvo in all files so that we can reduce the
# elf file sizes
# BSG_MANYCORE_LIB_OBJECTS  += bsg_printf.rvo

################################################################################
# Linker Flags
################################################################################

RISCV_LDFLAGS += -Wl,--defsym,bsg_group_size=$(_BSG_MACHINE_TILES)
RISCV_LDFLAGS += -Wl,--defsym,_bsg_elf_dram_size=$(BSG_ELF_DRAM_SIZE)
RISCV_LDFLAGS += -Wl,--defsym,_bsg_elf_vcache_size=$(BSG_ELF_VCACHE_MANYCORE_SIZE)
RISCV_LDFLAGS += -Wl,--defsym,_bsg_elf_stack_ptr=$(BSG_ELF_STACK_PTR)

RISCV_LDFLAGS += -nostdlib
RISCV_LDFLAGS += -march=$(ARCH_OP)
RISCV_LDFLAGS += -nostartfiles
RISCV_LDFLAGS += -ffast-math
RISCV_LDFLAGS += -lc
RISCV_LDFLAGS += -lm
RISCV_LDFLAGS += -lgcc

# TODO: temporary fix to solve this problem: https://stackoverflow.com/questions/56518056/risc-v-linker-throwing-sections-lma-overlap-error-despite-lmas-belonging-to-dif
RISCV_LDFLAGS += -Wl,--no-check-sections

################################################################################
# Linker Targets
################################################################################
# This builds a kernel.riscv binary for the current machine type and
# tile group size. KERNEL_OBJECTS can be used to include other object
# files in the linker. KERNEL_DEFAULT can be overridden by the user,
# but defaults to kernel.cpp

_LINK_HELP_STRING += "    kernel.riscv:\n"
_LINK_HELP_STRING += "        - Compile the RISC-V Manycore tensorlib Kernel.\n"
kernel.riscv: $(MACHINE_CRT_OBJ) main.rvo $(BSG_MANYCORE_LIB_OBJECTS) $(KERNEL_OBJECTS) $(basename $(KERNEL_DEFAULT)).rvo
	$(RISCV_LD) -T $(RISCV_LINK_SCRIPT) $^ $(RISCV_LDFLAGS) -o $@

kernel.link.clean:
	rm -rf *.riscv

.PRECIOUS: kernel.riscv %/kernel.riscv

_LINK_HELP_STRING += "\n"
_LINK_HELP_STRING += $(HELP_STRING)

HELP_STRING := $(_LINK_HELP_STRING)
