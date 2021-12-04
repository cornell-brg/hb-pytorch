#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

void prod_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_FLOAT_AND_INTS(iter.dtype(), "prod_hb", [&]() {
      offload_op_binary(iter, alpha_scalar.to<scalar_t>(), "tensorlib_prod");
      });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(prod_stub, &prod_kernel_hb);

}} // namespace at::native