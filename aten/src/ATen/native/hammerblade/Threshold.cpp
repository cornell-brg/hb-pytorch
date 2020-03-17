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

  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "threshold_hb", [&]() {
      offload_op_binary(
          iter, threshold_scalar, value_scalar, "tensorlib_threshold");
      });

}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(threshold_stub, &threshold_kernel_hb);

}} // namespace at::native
