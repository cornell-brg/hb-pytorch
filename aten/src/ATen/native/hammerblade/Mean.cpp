#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/hammerblade/Reduce.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

void mean_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "mean_hb", [&]() {
    scalar_t factor = scalar_t(iter.num_output_elements()) / scalar_t(iter.numel());
    binary_kernel_reduce(
      iter,
      MeanOps<scalar_t, scalar_t> {factor},
      scalar_t(0)
    );
  });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(mean_stub, &mean_kernel_hb);

}} // namespace at::native
