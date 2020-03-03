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
-include $(_REPO_ROOT)/hammerblade/environment.mk

################################################################################
# Include the host compilation rules. These define how to generate host object
# files
################################################################################
-include $(FRAGMENTS_PATH)/host/py_compile.mk

################################################################################
# Include the linker rules. These define how to generate the cosimulation binary
################################################################################
-include $(FRAGMENTS_PATH)/host/py_link.mk

################################################################################
# Include the analysis rules. These define how to generate analysis products
# like vanilla_operation_trace, etc.
################################################################################
-include $(FRAGMENTS_PATH)/analysis.mk

################################################################################
# Define rules for the default cosimulation execution.
################################################################################

ifeq ($(COSIM_DEBUG), 1)
COSIM_DEBUG_OPTS = +trace +vpdfile+$(HOST_TARGET).vpd
else
COSIM_DEBUG_OPTS = +NO_WAVES
endif

# ALIASES defines the outputs that are also generated when cosimulation is
# run. They are aliases for running $(HOST_TARGET).cosim.log. We use empty an
# make recipe for aliases for reasons described here:
# https://www.gnu.org/software/make/manual/html_node/Empty-Recipes.html
ALIASES = vanilla_stats.csv $(HOST_TARGET).vpd vanilla_operation_trace.csv
$(ALIASES): $(HOST_TARGET).cosim.log ;
$(HOST_TARGET).cosim.log: kernel.riscv $(HOST_TARGET).cosim 
	./$(HOST_TARGET).cosim +ntb_random_seed_automatic  \
		+c_args="$(PYTHON_SCRIPT) $(PYTHON_ARGS)" \
		$(COSIM_DEBUG_OPTS) | tee $@

cosim.clean: host.link.clean host.compile.clean
	rm -rf *.cosim{.daidir,.tmp,.log,} 64
	rm -rf vc_hdrs.h ucli.key
	rm -rf *.vpd *.vcs.log
	rm -rf $(HOST_TARGET)

.PHONY: $(HOST_TARGET).cosim.log cosim.clean

_HELP_STRING := "Rules from host/py_cosim.mk\n"
_HELP_STRING += "    $(HOST_TARGET).cosim.log: \n"
_HELP_STRING += "        - Run $(HOST_TARGET) on the tensorlib kernel\n"
_HELP_STRING += "        - Set COSIM_DEBUG=1 for trace and waveform dump\n"
_HELP_STRING += "\n"
_HELP_STRING += $(HELP_STRING)

HELP_STRING := $(_HELP_STRING)
