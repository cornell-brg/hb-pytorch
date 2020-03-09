#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Activation.h>

namespace at { namespace native {
namespace {


static void threshold_kernel_hb(
    TensorIterator& iter,
    Scalar threshold_scalar,
    Scalar value_scalar) {

    //TODO: implement the host code for threshold kernel
    //      you may refer to add_kernel_hb in
    //      aten/src/ATen/native/hammerblade/BinaryOpsKernel.cpp
    //      as an example

}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(threshold_stub, &threshold_kernel_hb);

}} // namespace at::native
