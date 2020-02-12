# setup brg sdh pytorch building env
echo "Setting up PyTorch building environment ... "
echo "Make sure you enabled devtoolset-8!"

# setup pytorch building options
export REL_WITH_DEB_INFO=1
export BUILD_TEST=0
export USE_CUDA=0
export USE_CUDNN=0
export USE_FBGEMM=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_DISTRIBUTED=0
export USE_OPENMP=0
export ATEN_THREADING=NATIVE

# setup cudalite runtime and pytorch kernel binary paths
if [ -z "$BRG_BSG_BLADERUNNER_DIR" ]
then
  export BSG_MANYCORE_DIR="<path-to-your-cudalite-cosim-runtime>"
else
  export BSG_MANYCORE_DIR=$BRG_BSG_BLADERUNNER_DIR/bsg_replicant/libraries
fi

if [ -z "$YODADA_BASELINE_DIR" ]
then
  export HB_KERNEL_DIR="<path-to-your-torch-kernel>"
else
  export HB_KERNEL_DIR=$YODADA_BASELINE_DIR/examples/torch/kernel.riscv
fi

echo "\$BSG_MANYCORE_DIR is set to $BSG_MANYCORE_DIR"
echo "\$HB_KERNEL_DIR is set to $HB_KERNEL_DIR"
