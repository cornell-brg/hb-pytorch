#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

void tanh_backward_hb(const Tensor & grad_output, const Tensor & output) {
   // AT_DISPATCH_FLOAT_TYPE_ONLY(grad_output.dtype(), output.dtype(), "tanh_backward_hb", [&]() {
        hb_offload_kernel(grad_output, output, "tensorlib_tanh_backward");
     //   });
}

} // anonymous namespace

//REGISTER_HAMMERBLADE_DISPATCH(tanh_backward_stub, &tanh_backward_kernel_hb);
//
}}  // namespace at::native
