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
# Paths / Environment Configuration
################################################################################
_REPO_ROOT ?= $(shell git rev-parse --show-toplevel)
CURRENT_PATH := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

-include $(_REPO_ROOT)/hammerblade/environment.mk

################################################################################
# Define BSG_MACHINE_PATH, the location of the Makefile.machine.include file
# that defines the machine to compile and simulate on. Using BSG_F1_DIR (which
# is set in environment.mk) uses the same machine as in bsg_replicant.
################################################################################

BSG_MACHINE_PATH=$(BSG_F1_DIR)

################################################################################
# Define any sources that should be used compiled during kernel compilation,
# including the source file with the kernel itself. kernel.riscv will
# be the name of the compiled RISC-V Binary for the Manycore
#
# Use KERNEL_INCLUDES to specify the path to directories that contain headers.
################################################################################

# C Libraries
KERNEL_CLIBRARIES   +=
# C++ Libraries
KERNEL_CXXLIBRARIES +=

KERNEL_INCLUDES     += -I$(CURRENT_PATH)/kernel

# Define the default kernel.cpp file. If KERNEL_DEFAULT is not defined it will
# be set to kernel.cpp in the same directory as this Makefile.
KERNEL_DEFAULT      := kernel/kernel.cpp

################################################################################
# Include the kernel build rules (This must be included after KERNEL_*LIBRARIES,
# KERNEL_DEFAULT, KERNEL_INCLUDES, etc)
################################################################################

-include $(FRAGMENTS_PATH)/kernel/cudalite.mk

################################################################################
# END OF KERNEL-SPECIFIC RULES / START OF HOST-SPECIFIC RULES
################################################################################

################################################################################
# Define the name of the python file to run
################################################################################
PYTHON_SCRIPT					= $(CURRENT_PATH)/pytest_runner.py
PYTHON_ARGS          ?=


################################################################################
# Define the $(HOST_TARGET), the name of the host executable to generate. The
# cosimulation host executable will be called
# $(HOST_TARGET).cosim. HOST_*SOURCES list the host files that should be
# compiled and linked into the executable.
################################################################################

HOST_TARGET         := test_loader
HOST_CSOURCES       :=
HOST_CXXSOURCES     := $(HOST_TARGET).cpp
HOST_INCLUDES       := -I$(CURRENT_PATH)

################################################################################
# Include the Cosimulation host build rules (This must be included after
# HOST_*SOURCES, HOST_TARGET, HOST_INCLUDES, etc)
################################################################################

-include $(FRAGMENTS_PATH)/host/py_cosim.mk

################################################################################
# Define the clean rules. clean calls the makefile-specific cleans, whereas
# users can add commands and dependencies to custom.clean.
################################################################################
version.clean:
	rm -rf kernel/*{.csv,.log,.rvo,.riscv,.vpd,.key,.png}
	rm -rf kernel/{stats,pc_stats}

custom.clean: version.clean

clean: cosim.clean analysis.clean cudalite.clean custom.clean

################################################################################
# Define overall-goals.
################################################################################

_HELP_STRING := "Makefile Rules\n"

_HELP_STRING += "    default: \n"
_HELP_STRING += "        - Run the pytest unit tests and generate all of the\n"
_HELP_STRING += "          analysis products\n"
_HELP_STRING += "        - Set PYTHON_SCRIPT=<path to python script>.\n"
_HELP_STRING += "        - Set PYTHON_ARGS=<argument to the script>.\n"
default: stats graphs pc_stats

_HELP_STRING += "    all: \n"
_HELP_STRING += "        - Same as default.\n"
all: default

.DEFAULT_GOAL = help
_HELP_STRING += "    help: \n"
_HELP_STRING += "        - Output a friendly help message.\n"
help:
	@echo -e $(HELP_STRING)

# Always re-run, if asked.
.PHONY: default analysis help

# These last three lines ensure that _HELP_STRING is appended to the top of
# whatever else comes before it.
_HELP_STRING += "\n"
_HELP_STRING += $(HELP_STRING)
HELP_STRING := $(_HELP_STRING)
