#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Fill.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

void fill_kernel_hb(TensorIterator& iter, Scalar value_scalar) {
  AT_DISPATCH_FLOAT_AND_INT_TYPE_ONLY(iter.dtype(), "fill_hb", [&]() {
      offload_op_nullary(iter, value_scalar.to<scalar_t>(), "tensorlib_fill");
      });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(fill_stub, &fill_kernel_hb);

}} // namespace at::native
