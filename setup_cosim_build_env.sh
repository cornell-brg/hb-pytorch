# setup brg sdh pytorch building env
echo "Setting up PyTorch building environment ... "
echo "Make sure you enabled devtoolset-6!"

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
export BSG_MANYCORE_DIR=/work/global/lc873/work/sdh/cosim/brg_bsg_bladerunner/bsg_replicant/libraries
export HB_KERNEL_DIR=/work/global/lc873/work/sdh/cosim/baseline/examples/torch/kernel.riscv
