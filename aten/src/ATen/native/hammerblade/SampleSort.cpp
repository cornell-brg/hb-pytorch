#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

Tensor sample_sort_hb(const Tensor &inp, const int64_t nproc, const int64_t sr) {
    
    auto sorted_ = at::zeros(inp.sizes(),inp.options());
    auto sample_keys = at::zeros(nproc*sr,inp.options());
    auto sorted_keys = at::zeros(nproc*sr,inp.options());
    auto splitters = at::zeros(nproc,inp.options());
    auto buck_sizes = at::zeros(nproc,inp.options());
    auto bucket_id = at::zeros(inp.sizes(),inp.options());
    int32_t nproc_i32 = safe_downcast<int32_t, int64_t>(nproc);
    int32_t sr_i32 = safe_downcast<int32_t, int64_t>(sr);
    
    hb_offload_kernel(sorted_,inp,sample_keys,sorted_keys,splitters,buck_sizes,bucket_id,nproc_i32,sr_i32,"tensorlib_samplesort");

    return sorted_;
}

}} // namespace at::native