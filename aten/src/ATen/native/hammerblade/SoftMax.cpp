#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

Tensor log_softmax_hb(const Tensor& input_, const int64_t dim_,
                      const bool half_to_float) {
  Tensor output = at::empty(input_.sizes(), input_.options());
  TORCH_CHECK(false, "log_softmax_hb not implemented.");
  return output;
}

}} // at::native
