#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {


void eq_kernel_hb(TensorIterator& iter) {
  if(iter.dtype(1) == kFloat) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Bool, iter.dtype(), "eq_hb", [&]() {
      offload_op_binary(iter, "tensorlib_eq_Float");
    });
  } else if (iter.dtype(1) == kInt) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Bool, iter.dtype(), "eq_hb", [&]() {
      offload_op_binary(iter, "tensorlib_eq_Int");
    });
  }
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(eq_stub, &eq_kernel_hb);

}} // namespace at::native
