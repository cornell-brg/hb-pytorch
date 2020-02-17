
_REPO_ROOT           := $(shell git rev-parse --show-toplevel)

# Check if we are running the gloablly installed COSIM ob brg-vip
ifdef BRG_BSG_BLADERUNNER_DIR
-include $(BRG_BSG_BLADERUNNER_DIR)/project.mk
endif

# Check if we are running inside of the BSG Bladerunner repository by searching
# for project.mk. If project.mk is found, then we are and we should use
# that to define BASEJUMP_STL_DIR, and BSG_MANYCORE_DIR and
# override previous settings. Warn, to provide feedback.
ifneq ("$(wildcard $(_REPO_ROOT)/../project.mk)","")

# Override BASEJUMP_STL_DIR, and BSG_MANYCORE_DIR.
# If they were previously set, warn if there is a mismatch between the
# new value and the old (temporary). Using := is critical here since
# it assigns the value of the variable immediately.

# If BASEJUMP_STL_DIR is set, save it to a temporary variable to check
# against what is set by Bladrunner and undefine it
ifdef BASEJUMP_STL_DIR
_BASEJUMP_STL_DIR := $(BASEJUMP_STL_DIR)
undefine BASEJUMP_STL_DIR
endif

# If BSG_MANYCORE_DIR is set, save it to a temporary variable to check
# against what is set by Bladrunner and undefine it
ifdef BSG_MANYCORE_DIR
_BSG_MANYCORE_DIR := $(BSG_MANYCORE_DIR)
undefine BSG_MANYCORE_DIR
endif

# Include project.mk from Bladerunner. This will override
# BASEJUMP_STL_DIR, and BSG_MANYCORE_DIR
include $(_REPO_ROOT)/../project.mk

ifdef _BASEJUMP_STL_DIR
ifneq ($(_BASEJUMP_STL_DIR), $(BASEJUMP_STL_DIR))
$(warning $(shell echo -e "$(ORANGE)BSG MAKE WARN: Overriding BASEJUMP_STL_DIR environment variable with Bladerunner defaults.$(NC)"))
$(warning $(shell echo -e "$(ORANGE)BSG MAKE WARN: BASEJUMP_STL_DIR=$(BASEJUMP_STL_DIR)$(NC)"))
endif # Matches: ifneq ($(_BASEJUMP_STL_DIR), $(BASEJUMP_STL_DIR))
endif # Matches: ifdef _BASEJUMP_STL_DIR
# Undefine the temporary variable to prevent its use
undefine _BASEJUMP_STL_DIR

ifdef _BSG_MANYCORE_DIR
ifneq ($(_BSG_MANYCORE_DIR), $(BSG_MANYCORE_DIR))
$(warning $(shell echo -e "$(ORANGE)BSG MAKE WARN: Overriding BSG_MANYCORE_DIR environment variable with Bladerunner defaults.$(NC)"))
$(warning $(shell echo -e "$(ORANGE)BSG MAKE WARN: BSG_MANYCORE_DIR=$(BSG_MANYCORE_DIR)$(NC)"))
endif # Matches: ifneq ($(_BSG_MANYCORE_DIR), $(BSG_MANYCORE_DIR))
endif # Matches: ifdef _BSG_MANYCORE_DIR
# Undefine the temporary variable to prevent its use
undefine _BSG_MANYCORE_DIR
endif # Matches: ifneq ("$(wildcard $(CL_DIR)/../project.mk)","")

# If BASEJUMP_STL_DIR is not defined at this point, raise an error.
ifndef BASEJUMP_STL_DIR
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: BASEJUMP_STL_DIR environment variable undefined. Defining is not recommended. Are you running from within Bladerunner?$(NC)"))
endif

# If BSG_MANYCORE_DIR is not defined at this point, raise an error.
ifndef BSG_MANYCORE_DIR
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: BSG_MANYCORE_DIR environment variable undefined. Defining is not recommended. Are you running from within Bladerunner?$(NC)"))
endif

# TODO: Check if exists
RISCV_BIN_DIR=$(BSG_MANYCORE_DIR)/software/riscv-tools/riscv-install/bin/

FRAGMENTS_PATH=$(_REPO_ROOT)/fragments

