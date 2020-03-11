#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {


static void abs_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "abs_hb", [&]() {
      offload_op_unary(iter, Scalar(), "tensorlib_abs");
      });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(abs_stub, &abs_kernel_hb);

}} // namespace at::native
