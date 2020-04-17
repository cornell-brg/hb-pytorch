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
    device_args.push_back(create_device_tensor(output, device_ptrs));
    device_args.push_back(create_device_scalar(n));
    device_args.push_back(create_device_scalar(m));
    c10::hammerblade::offload_kernel("tensorlib_eye", device_args);
    cleanup_device(device_args, device_ptrs);
    return output;
  }

}}

