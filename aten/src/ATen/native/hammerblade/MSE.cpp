#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

static void mse_kernel_hb(TensorIterator& iter) {

  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "mse_hb", [&](){
        offload_op_binary(iter, "tensorlib_mse");
        });

}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(mse_stub, &mse_kernel_hb);

}} // namespace at::native
