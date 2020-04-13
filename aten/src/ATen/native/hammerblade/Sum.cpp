#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

void sum_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "sum_hb", [&]() {
      if(iter.num_output_elements() == 1) {
        offload_op_unary(iter, "tensorlib_sum_simple");
      } else {
        offload_iterator_reduce_op_impl<scalar_t>(iter, "tensorlib_sum");
      }
  });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(sum_stub, &sum_kernel_hb);

}} // namespace at::native
