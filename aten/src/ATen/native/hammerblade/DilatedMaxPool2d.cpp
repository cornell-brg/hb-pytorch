#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {
namespace { // anonumous
} // anonymous

std::tuple<Tensor, Tensor> max_pool2d_with_indices_hb(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode)
{
  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));
  //max_pool2d_with_indices_out_cpu_template(
  //  output,
  //  indices,
  //  input,
  //  kernel_size,
  //  stride,
  //  padding,
  //  dilation,
  //  ceil_mode);
  TORCH_CHECK(false, "Implement HB maxpool2d.");
  return std::tuple<Tensor, Tensor>(output, indices);
}

}} // at::native
