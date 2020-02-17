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

################################################################################
# Paths
################################################################################
_REPO_ROOT ?= $(shell git rev-parse --show-toplevel)
-include $(_REPO_ROOT)/environment.mk

################################################################################
# Tools
################################################################################

RISCV_GCC     ?= $(RISCV_BIN_DIR)/riscv32-unknown-elf-dramfs-gcc
RISCV_GXX     ?= $(RISCV_BIN_DIR)/riscv32-unknown-elf-dramfs-g++
RISCV_ELF2HEX ?= LD_LIBRARY_PATH=$(RISCV_BIN_DIR)/../lib $(RISCV_BIN_DIR)/elf2hex
RISCV_OBJCOPY ?= $(RISCV_BIN_DIR)/riscv32-unknown-elf-dramfs-objcopy
RISCV_AR      ?= $(RISCV_BIN_DIR)/riscv32-unknown-elf-dramfs-ar
RISCV_LD      ?= $(RISCV_GCC)
RISCV_LINK    ?= $(RISCV_GCC) -t -T $(LINK_SCRIPT) $(RISCV_LDFLAGS)
RISCV_OBJDUMP ?= $(RISCV_BIN_DIR)/riscv32-unknown-elf-dramfs-objdump
