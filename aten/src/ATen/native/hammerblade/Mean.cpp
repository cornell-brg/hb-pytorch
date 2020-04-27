#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

void mean_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "mean_hb", [&]() {
    offload_iterator_reduce_op_impl<scalar_t>(iter, "tensorlib_mean");
  });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(mean_stub, &mean_kernel_hb);

}} // namespace at::native
