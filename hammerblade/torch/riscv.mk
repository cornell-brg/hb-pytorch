.DEFAULT = help

BSG_MANYCORE_DIR := $(BRG_BSG_BLADERUNNER_DIR)/bsg_manycore
COSIM_PYTHON_DIR := $(BRG_BSG_BLADERUNNER_DIR)/bsg_replicant/testbenches/python
KERNEL_DIR       := $(shell git rev-parse --show-toplevel)/hammerblade/torch/kernel
SCRIPT_DIR       := $(shell git rev-parse --show-toplevel)/hammerblade/torch/
SCRIPT           := pytest_runner

# bsg_replicant detail: test_loader is fast and lean version of
# simulation executable. test_loader.debug is waveform enabled version.
COSIM_EXE :=
ifeq ($(WAVES),1)
  COSIM_EXE := $(COSIM_PYTHON_DIR)/test_loader.debug
else
  COSIM_EXE := $(COSIM_PYTHON_DIR)/test_loader
endif

COSIM_DEBUG :=
ifeq ($(TRACE),1)
  # bsg_replicant detail: +trace enables execution trace
  COSIM_DEBUG += +trace
endif

# Disable CADENV to resolve related errors in BSG_MANYCORE. We don't need
# CADENV here, as we only import RISC-V GCC buildefs and are not building
# CAD related things.
export IGNORE_CADENV := 1

help:
	@echo "regression:"
	@echo "    Run pytest regression on $(SCRIPT_DIR)/tests direcotry"
	@echo ""
	@echo "kernel.riscv:"
	@echo "    Build the device binary with suite of kernel implementations"
	@echo "    in $(KERNEL_DIR)."
	@echo "kernel.dis:"
	@echo "    Output disassembly of kernel binary to standard out."
	@echo "Flags:"
	@echo "     TRACE=1: dumps execution trace to vanilla.log"
	@echo "     WAVES=1: enable waveform dump"

.PHONY: regression
regression: kernel.riscv $(COSIM_EXE)
	$(COSIM_EXE) +ntb_random_seed_automatic \
		+c_args="$(SCRIPT_DIR) $(SCRIPT)" $(COSIM_DEBUG) \
		2>&1 | tee regression.log

$(COSIM_EXE):
	$(MAKE) -C $(COSIM_PYTHON_DIR) $@

ifeq ($(DEBUG),1)
  RISCV_GXX_EXTRA_OPTS += -g
  RISCV_LINK_OPTS += -g
endif

# Include BSG Manycore's builddefs
include $(BSG_MANYCORE_DIR)/software/mk/Makefile.master

INCS := -I$(KERNEL_DIR)

KERNEL_CSRCS   := $(notdir $(wildcard $(KERNEL_DIR)/*.c))
KERNEL_CPPSRCS := $(notdir $(wildcard $(KERNEL_DIR)/*.cpp))
KERNEL_OBJS    := $(patsubst %.c,%.o,$(KERNEL_CSRCS)) \
                  $(patsubst %.cpp,%.o,$(KERNEL_CPPSRCS))

$(KERNEL_CSRCS): %.c : $(KERNEL_DIR)/%.c
	cp $^ $@

$(KERNEL_CPPSRCS): %.cpp : $(KERNEL_DIR)/%.cpp
	cp $^ $@

kernel.riscv: LINK_SCRIPT=$(SCRIPT_DIR)/hb_pytorch_link.ld
kernel.riscv: $(SPMD_COMMON_OBJECTS) $(BSG_MANYCORE_LIB) crt.o
kernel.riscv: $(KERNEL_OBJS)
	$(RISCV_LINK) $(KERNEL_OBJS) $(SPMD_COMMON_OBJECTS) \
		-L. -l:$(BSG_MANYCORE_LIB) -o $@ $(filter-out -nostdlib,$(RISCV_LINK_OPTS))

clean:
	-rm -rf stack.info.* *.log *.csv ucli.key
	-rm -rf *.o *.riscv
	-rm -rf *.rvo *.vpd
	-rm -rf $(BSG_MANYCORE_LIB) $(KERNEL_CSRCS) $(KERNEL_CPPSRCS)
	-rm -rf $(KERNEL_DIR)/*.rvo
	-rm -rf $(KERNEL_DIR)/*.gcc.log
