#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

Tensor cdist_hb(const Tensor& x1, const Tensor& x2, double p, c10::optional<int64_t> compute_mode) {

    TORCH_CHECK(x1.is_hammerblade(), "cdist hb: expected 'x1' to be a HammerBlade tensor");
    TORCH_CHECK(x2.is_hammerblade(), "cdist hb: expected 'x2' to be a HammerBlade tensor");

    TORCH_CHECK(x1.dim() == 2 && x2.dim() == 2, "2D matrices expected, got ", x1.dim(), " and ", x2.dim(), " tensors");
    TORCH_CHECK(x1.size(1) == x2.size(1), "2nd dimension, observations should be equal! Got ", x1.size(1), " and ", x2.size(1));

    Tensor result = at::empty({x1.size(0), x2.size(0)}, x1.options());

    hb_offload_kernel(result, x1, x2, "tensorlib_cdist");
    return result;
}

} // anonymous namespace

}} // namespace at::native