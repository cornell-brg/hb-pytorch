#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>
#include <cmath>

namespace at { namespace native {

Tensor& arange_hb_out(Tensor& result, Scalar start, Scalar end, Scalar step) {

  auto xstart = start.to<int32_t>();
  auto xend = end.to<int32_t>();
  auto xstep = step.to<int32_t>();

  double size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>()) / step.to<double>());

  TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
  TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) &&
             std::isfinite(static_cast<double>(xend)),
             "unsupported range: ", xstart, " -> ", xend);
  TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
             "upper bound and larger bound inconsistent with step sign");

  TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
             "invalid size, possible overflow?");

  int64_t size = static_cast<int64_t>(size_d);
  int64_t numel = result.numel();

  if (numel != size) {
    if(numel > 0){
      TORCH_WARN("The number of elements in the out tensor of shape ", result.sizes(),
                  " is ", numel, " which does not match the computed number of elements ", size,
                  ". Note that this may occur as a result of rounding error. "
                  "The out tensor will be resized to a tensor of shape (", size, ",).");
    }
    result.resize_({size});
  }
  result = result.to(at::kInt);
  int32_t size_32 = (int32_t)size;
  hb_offload_kernel(result, xstart, xstep, size_32, "tensorlib_arange");
  return result;
}
  
}}  
