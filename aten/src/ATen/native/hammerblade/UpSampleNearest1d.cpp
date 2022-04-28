#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/hammerblade/OffloadDef.h>
#include <ATen/native/hammerblade/OffloadUtils.h>

namespace at { namespace native {

Tensor& upsample_nearest1d_out_hb(Tensor& output, const Tensor& input, IntArrayRef output_size) {
  // Add code here
  return output;
}

Tensor upsample_nearest1d_hb(const Tensor& input, IntArrayRef output_size) {
//  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "upsample_nearest1d_hb", [&]() {
//      offload_op_unary(iter, "tensorlib_upsample_nearest1d");
//      });
  auto output = at::empty({0}, input.options());
  upsample_nearest1d_out_hb(output, input, output_size);
  return output;
}

}} // namespace at::native
