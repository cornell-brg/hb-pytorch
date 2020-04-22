#!/usr/bin/bash -x

COSIM_PYTHON_EXE=$BRG_BSG_BLADERUNNER_DIR/bsg_replicant/testbenches/pytorch/test_loader

if [[ ! -f $COSIM_PYTHON_EXE ]]
then
  echo "Error: cannot find the cosim executable $COSIM_PYTHON_EXE." \
       "Make sure to run \"source setup_cosim_build_env.sh\" in $TOPLEVEL" 1>&2
  exit 1
fi

# COSIM_PYTHON_EXE is the VCS executable in bladerunner.
eval "$COSIM_PYTHON_EXE +ntb_random_seed_automatic +trace +c_args=\""$@"\"" \
     "| grep -v \": instantiating\|\[.*_PROFILER\]\""
