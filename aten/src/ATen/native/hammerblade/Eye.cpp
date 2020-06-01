#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

  Tensor& eye_out_hb(Tensor& output, long n, long m) {
    std::vector<eva_t> device_args;
    std::vector<eva_t> device_ptrs;
    if(m == -1) {
      m = n;
    }
    output.resize_({n,m});

    uint32_t m_u32 = safe_downcast<uint32_t, long>(m);
    uint32_t n_u32 = safe_downcast<uint32_t, long>(n);

    hb_offload_kernel(output, n_u32, m_u32, "tensorlib_eye");
    return output;
  }

}}

