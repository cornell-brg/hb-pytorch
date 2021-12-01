#!/usr/bin/bash -x

COSIM_PYTHON_EXE=$BRG_BSG_BLADERUNNER_DIR/bsg_replicant/machines/pod_X1Y1_ruche_X16Y8_hbm_one_pseudo_channel/bigblade-vcs/exec/simv

if [[ ! -f $COSIM_PYTHON_EXE ]]
then
  echo "Error: cannot find the cosim executable $COSIM_PYTHON_EXE." \
       "Make sure to run \"source setup_cosim_build_env.sh\" in $TOPLEVEL" 1>&2
  exit 1
fi

# COSIM_PYTHON_EXE is the VCS executable in bladerunner.
eval "$COSIM_PYTHON_EXE +ntb_random_seed_automatic +c_args=\""$@"\"" \
     "+c_path=/work/shared/common/project_build/bigblade-6.4/bsg_replicant/examples/python/test_loader/main.so " \
     "| grep -v \": instantiating\|\[.*_PROFILER\]\""
