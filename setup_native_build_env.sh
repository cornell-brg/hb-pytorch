# setup brg sdh pytorch building env
echo "  Setting up PyTorch building environment ... for native evaluation!"
echo "  Make sure you enabled devtoolset-8!"
echo ""

# setup pytorch building options
export BUILD_TEST=0
export USE_HB=0
export BLAS=OpenBLAS
export OpenBLAS_HOME=<YOUR_OpenBLAS_HOME_PATH>
export USE_MKL=0
export USE_MKLDNN=0
export USE_CUDA=0
export USE_CUDNN=0
export USE_FBGEMM=0
export USE_NNPACK=0
export USE_QNNPACK=0

# Use gold if it's available for faster linking.
if which gold >/dev/null 2>&1 ; then
    export CFLAGS='-fuse-ld=gold'
fi

# make sure real bsg_manycore_cuda library is not exported
unset BSG_MANYCORE_DIR
unset HB_KERNEL_DIR
