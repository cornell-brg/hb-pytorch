#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Activation.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {


static void threshold_kernel_hb(
    TensorIterator& iter,
    Scalar threshold_scalar,
    Scalar value_scalar) {

    //TORCH_CHECK(false, "threshold_kernel_hb not implemented");
    //TODO: implement the host code for threshold kernel
    //      you may refer to add_kernel_hb in
    //      aten/src/ATen/native/hammerblade/AddSub.cpp
    //      as an example
    AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "threshold_hb", [&]() {
        offload_op_binary(iter, threshold_scalar, value_scalar, "tensorlib_threshold");
        });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(threshold_stub, &threshold_kernel_hb);

}} // namespace at::native
