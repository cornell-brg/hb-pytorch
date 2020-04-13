#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {


static void round_kernel_hb(TensorIterator& iter) {  
  // TODO: replace with 
  //AT_DISPATCH_FLOATING_TYPES //double, float
  //AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES // (complex & not) double, float

  AT_DISPATCH_FLOAT_TYPE_ONLY(
    iter.dtype(), //not used, but here anyway
    "round_hb", //binds name round to hammerblade
      [&]() {//anonymous function that captures & passes all variables (ie iter) by reference ([&])
        offload_op_unary(iter, "tensorlib_round");
      });
}

} // anonymous namespace

//when we call python's round_stub on device, use round_stubround_kernel_hb
REGISTER_HAMMERBLADE_DISPATCH(round_stub, &round_kernel_hb);

}} // namespace at::native
