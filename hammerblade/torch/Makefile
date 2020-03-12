.DEFAULT = help

COSIM_EXE        := $(BRG_BSG_BLADERUNNER_DIR)/bsg_replicant/testbenches/python/test_loader
BSG_MANYCORE_DIR := $(BRG_BSG_BLADERUNNER_DIR)/bsg_manycore
KERNEL_DIR       := $(shell git rev-parse --show-toplevel)/hammerblade/torch/kernel
SCRIPT_DIR       := $(shell git rev-parse --show-toplevel)/hammerblade/torch/
SCRIPT           := pytest_runner

ifeq ($(wildcard $(COSIM_EXE)),)
  $(error Error: COSIM executable missing in \
    $(BRG_BSG_BLADERUNNER_DIR)/bsg_replicant/testbenches/python. \
    Make sure to run setup_cosim_build_env.sh script and build \
    PyTorch for COSIM before running tests in COSIM)
endif

help:
	echo ""

regression: kernel.riscv
	$(COSIM_EXE) +ntb_random_seed_automatic \
		+c_args="$(SCRIPT_DIR) $(SCRIPT)" +trace \
		2>&1 | tee regression.log

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

kernel.riscv: $(SPMD_COMMON_OBJECTS) $(BSG_MANYCORE_LIB) crt.o
kernel.riscv: $(KERNEL_OBJS)
	$(RISCV_LINK) $(KERNEL_OBJS) $(SPMD_COMMON_OBJECTS) \
		-L. -l:$(BSG_MANYCORE_LIB) -o $@ $(RISCV_LINK_OPTS)

clean:
	-rm -rf stack.info.* *.log *.csv ucli.key
	-rm -rf *.o *.riscv
	-rm -rf $(BSG_MANYCORE_LIB) $(KERNEL_CSRCS) $(KERNEL_CPPSRCS)
