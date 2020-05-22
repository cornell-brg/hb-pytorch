#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

static void floor_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "floor_hb", [&]() {
      offload_op_unary(iter, "tensorlib_floor");
      });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(floor_stub, &floor_kernel_hb);

}} // namespace at::native
