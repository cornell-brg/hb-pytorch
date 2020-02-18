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

################################################################################
# Analysis rules 
################################################################################
_HELP_STRING := "Rules from analysis.mk\n"
_HELP_STRING += "    kernel.dis:\n"
_HELP_STRING += "        - Disassemble RISC-V binary of the tensorlib kernel\n"
%.dis: %.riscv
	$(RISCV_OBJDUMP) -M numeric --disassemble-all -S $< > $@

_HELP_STRING += "    stats:\n"
_HELP_STRING += "        - Run the Vanilla Stats Parser on the output of $(HOST_TARGET).cosim\n"
_HELP_STRING += "          run on the tensorlib kernel to generate statistics\n"
stats: vanilla_stats.csv
	python3 $(BSG_MANYCORE_DIR)/software/py/vanilla_stats_parser.py --tile --tile_group

_HELP_STRING += "    graphs:\n"
_HELP_STRING += "        - Run the Operation Trace Parser on the output of $(HOST_TARGET).cosim\n"
_HELP_STRING += "          run on the tensorlib kernel to generate the\n"
_HELP_STRING += "          abstract and detailed profiling graphs\n"
graphs: blood_abstract.png blood_detailed.png

blood_detailed.png: vanilla_operation_trace.csv vanilla_stats.csv
	python3 $(BSG_MANYCORE_DIR)/software/py/blood_graph.py --input vanilla_operation_trace.csv --timing-stats vanilla_stats.csv --generate-key

blood_abstract.png: vanilla_operation_trace.csv vanilla_stats.csv
	python3 $(BSG_MANYCORE_DIR)/software/py/blood_graph.py --input vanilla_operation_trace.csv --timing-stats vanilla_stats.csv --generate-key --abstract

_HELP_STRING += "    pc_stats:\n"
_HELP_STRING += "        - Run the Program Counter Histogram utility on the output of\n"
_HELP_STRING += "          $(HOST_TARGET).cosim run on the tensorlib kernel to \n"
_HELP_STRING += "          generate the Program Counter Histogram\n"
pc_stats: vanilla_operation_trace.csv
	python3 $(BSG_MANYCORE_DIR)/software/py/vanilla_pc_histogram.py --dim-x $(_BSG_MACHINE_TILES_X) --dim-y $(_BSG_MACHINE_TILES_Y) --tile --input $<

analysis.clean:
	rm -rf *.dis
	rm -rf vanilla_stats.csv vanilla_operation_trace.csv vanilla.log vcache_non_blocking_stats.log vcache_blocking_stats.log
	rm -rf stats pc_stats
	rm -rf blood_abstract.png blood_detailed.png
	rm -rf key_abstract.png key_detailed.png

.PRECIOUS: %.png %/blood_detailed.png %/blood_abstract.png


_HELP_STRING += "\n"
_HELP_STRING += $(HELP_STRING)

HELP_STRING := $(_HELP_STRING)

