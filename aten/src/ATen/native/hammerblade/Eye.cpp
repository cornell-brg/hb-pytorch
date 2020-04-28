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
    hb_offload_kernel(output, n, m, "tensorlib_eye"); 
    return output;
  }

}}

