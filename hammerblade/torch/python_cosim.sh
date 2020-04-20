#!/usr/bin/bash -x

TOPLEVEL=$(git rev-parse --show-toplevel)
COSIM_PYTHON_DIR=$BRG_BSG_BLADERUNNER_DIR/bsg_replicant/testbenches/pytorch
COSIM_PYTHON_EXE=$COSIM_PYTHON_DIR/test_loader

if [[ ! -f $COSIM_PYTHON_EXE ]]
then
	echo "Error: cannot find the cosim executable $COSIM_PYTHON_EXE." \
		   "Make sure to run \"source setup_cosim_build_env.sh\" in $TOPLEVEL" 1>&2
	exit 1
fi

# Absolute path to first command line argument is the script
# to execute
SCRIPT_PATH=$(realpath $1)

COSIM_CMD=($COSIM_PYTHON_EXE +ntb_random_seed_automatic +c_args=\"$SCRIPT_PATH\")
COSIM_CMD=("${COSIM_CMD[@]}" "|" grep -v \": instantiating\")

# COSIM_PYTHON_EXE is the VCS executable in bladerunner.
eval ${COSIM_CMD[@]}
