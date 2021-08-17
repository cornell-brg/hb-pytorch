#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

static void pow_tensor_scalar_kernel_hb(TensorIterator& iter, Scalar exp_scalar) {


  // NOTE: Complex inputs are not supported
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "pow_tensor_scalar_kernel_hb", [&](){
        offload_op_unary(iter, exp_scalar.to<scalar_t>(), "tensorlib_pow_tensor_scalar");
        });

}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel_hb);

}} // namespace at::native
