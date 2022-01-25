#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {


void bitwise_not_kernel_hb(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Bool, iter.dtype(), "bitwise_not_bool_hb", [&]() {
      offload_op_unary(iter, "tensorlib_bitwise_not_bool");
    });
  } else {
    AT_DISPATCH_INT_TYPE_ONLY(iter.dtype(), "bitwise_not_int_hb", [&]() {
      offload_op_unary(iter, "tensorlib_bitwise_not_int");
    }); 
  }
}

}

REGISTER_HAMMERBLADE_DISPATCH(bitwise_not_stub, &bitwise_not_kernel_hb);

}} // namespace at::native
