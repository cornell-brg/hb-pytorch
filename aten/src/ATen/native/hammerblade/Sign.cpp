#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

static void sign_kernel_hb(TensorIterator& iter) {

  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "sign_hb", [&](){
        offload_op_unary(iter, "tensorlib_sign");
        });

}

}

REGISTER_HAMMERBLADE_DISPATCH(sign_stub, &sign_kernel_hb);

}}
