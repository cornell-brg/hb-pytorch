#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {
namespace {

static void addcdiv_kernel_hb(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "addcdiv_hb", [&]() {
    offload_op_ternary(iter, value.to<scalar_t>(), "tensorlib_addcdiv");
  });
}


} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(addcdiv_stub, &addcdiv_kernel_hb);

} // namespace native
} // namespace at
