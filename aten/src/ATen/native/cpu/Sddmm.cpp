#include <algorithm>
#include <vector>

namespace at {
namespace native {

template <typename scalar_t>
void inline sddmm_kernel(
  const Tensor& a,
  const Tensor& b,
  const Tensor& c,
  Tensor& out_tensor
){
  // do something here
}

Tensor sddmm_cpu(const Tensor& a, const Tensor& b, const Tensor& c, Tensor& out) {
  AT_DISPATCH_ALL_TYPES(a.scalar_type(), "sddmm_cpu", [&] {
    sddmm_kernel<scalar_t>(
      a, b, c, out
    );
  });

  return out;
}


}} // namespace at::native
