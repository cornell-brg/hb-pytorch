# setup brg sdh pytorch building env
echo "  Setting up PyTorch building environment ... "
echo "  Make sure you enabled devtoolset-8!"
echo "  Make sure correct Python environemnt is set!"
echo ""

# setup pytorch building options
export REL_WITH_DEB_INFO=1
export BUILD_TEST=0
export USE_MKL=0
export USE_MKLDNN=0
export USE_CUDA=0
export USE_CUDNN=0
export USE_FBGEMM=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_DISTRIBUTED=0
export OMP_NUM_THREADS=1

# Use gold if it's available for faster linking.
if which gold >/dev/null 2>&1 ; then
    export CFLAGS='-fuse-ld=gold'
fi

# get current directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

echo "  hb-pytorch lives in $DIR"

export HB_SILICON_V0=1

# setup cudalite runtime and pytorch kernel binary paths
export BSG_MANYCORE_INCLUDE=/usr/include
export BSG_MANYCORE_LDPATH=/usr/lib64

export HB_KERNEL_DIR=$DIR/hammerblade/torch

echo "  \$BRG_BSG_BLADERUNNER_DIR is set to $BRG_BSG_BLADERUNNER_DIR"
echo "  \$BSG_MANYCORE_INCLUDE is set to $BSG_MANYCORE_INCLUDE"
echo "  \$BSG_MANYCORE_LDPATH is set to $BSG_MANYCORE_LDPATH"
echo "  \$HB_KERNEL_DIR is set to $HB_KERNEL_DIR"
echo ""
echo "  Done!"
echo ""
