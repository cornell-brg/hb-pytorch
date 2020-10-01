#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

void and_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_FLOAT_AND_INTS(iter.dtype(), "and_hb", [&]() {
      offload_iterator_reduce_op_impl<scalar_t>(iter, "tensorlib_and");
      });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(and_stub, &and_kernel_hb);


}} // namespace at::native
