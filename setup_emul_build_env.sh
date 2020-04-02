# setup brg sdh pytorch building env
echo "  Setting up PyTorch building environment ... with EMULATION!"
echo "  Make sure you enabled devtoolset-8!"
echo ""

# setup pytorch building options
export DEBUG=1
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
export CFLAGS='-fuse-ld=ld'

# enable emulation layer building
export USE_HB_EMUL=1

# make sure real bsg_manycore_cuda library is not exported
unset BSG_MANYCORE_DIR
unset HB_KERNEL_DIR
