#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

namespace {

void not_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_INTS_ONLY(iter.dtype(), "not_hb", [&]() {
    std::cout << "host code dispatching";
    offload_op_unary(iter, "tensorlib_not");
  });
}

} // anonymous namespace
REGISTER_HAMMERBLADE_DISPATCH(bitwise_not_stub, &not_kernel_hb);
}} // namespace at::native
