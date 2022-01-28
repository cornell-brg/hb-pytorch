#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {


void ne_kernel_hb(TensorIterator& iter) {
  if(iter.dtype(1) == kFloat) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Bool, iter.dtype(), "ne_hb", [&]() {
      offload_op_binary(iter, "tensorlib_ne_Float");
    });
  } else if (iter.dtype(1) == kInt) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Bool, iter.dtype(), "ne_hb", [&]() {
      offload_op_binary(iter, "tensorlib_ne_Int");
    });
  }
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(ne_stub, &ne_kernel_hb);

}} // namespace at::native
