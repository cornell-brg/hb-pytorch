#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {


static void sigmoid_backward_kernel_hb(TensorIterator& iter) {

  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "sigmoid_backward_hb", [&]() {
    offload_op_binary(iter, "tensorlib_sigmoid_backward");
  });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(sigmoid_backward_stub, &sigmoid_backward_kernel_hb);

}} // namespace at::native
