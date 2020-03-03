################################################################################
# BSG Manycore Machine Configuration
################################################################################

# Import configuration from BSG_MACHINE_PATH.
-include $(BSG_MACHINE_PATH)/Makefile.machine.include

# Set architecture variables (if not already set)
bsg_global_X ?= $(BSG_MACHINE_GLOBAL_X)
bsg_global_Y ?= $(BSG_MACHINE_GLOBAL_Y)

_BSG_MACHINE_TILES_X ?= $(BSG_MACHINE_GLOBAL_X)
_BSG_MACHINE_TILES_Y ?= $(shell expr $(BSG_MACHINE_GLOBAL_Y) - 1)
_BSG_MACHINE_TILES   ?= $(shell expr $(_BSG_MACHINE_TILES_X) \* $(_BSG_MACHINE_TILES_Y))

