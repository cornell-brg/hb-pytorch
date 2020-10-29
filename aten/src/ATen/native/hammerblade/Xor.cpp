#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

void xor_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_INTS_ONLY(iter.dtype(), "xor_hb", [&]() {
      offload_op_binary(iter, "tensorlib_xor");
      });
}

} // anonymous namespace
REGISTER_HAMMERBLADE_DISPATCH(bitwise_xor_stub, &xor_kernel_hb);
}} // namespace at::native
