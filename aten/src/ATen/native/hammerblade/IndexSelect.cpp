#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

Tensor index_select_hb(const Tensor& self, int64_t dim, const Tensor& index) {

  auto ndim = self.dim();
  if (ndim == 0) {
    AT_INDEX_ERROR("index_select() cannot be applied to a 0-dim tensor.");
  }
  if (!(index.dim() == 1 && index.dtype() == at::kLong)) {
    AT_INDEX_ERROR("index_select() argument index must be 1-D long-tensor.");
  }

  dim = maybe_wrap_dim(dim, ndim);
  auto size = self.size(dim);
  auto new_sizes = self.sizes().vec();
  new_sizes[dim] = index.size(0);
  auto numel = index.numel();
  auto result = at::empty(new_sizes, self.options());

  // fast path -- we can simply use memcpy
  if (dim == 0 && self.is_contiguous()) {
    auto index_int = index.to(at::kInt).contiguous();

    hb_offload_kernel(self, result, index_int, "tensorlib_index_select");
  } else {
    // slow path
    auto index_cpu = index.cpu();
    int64_t* index_data = index_cpu.data_ptr<int64_t>();

    for (size_t i=0; i<numel; i++) {
      auto dst = at::select(result, dim, i);
      auto src = at::select(self, dim, index_data[i]);
      at::native::copy_(dst, src);
    }
  }

  return result;
}

}} // at::native
