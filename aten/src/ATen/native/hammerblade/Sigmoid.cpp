#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

static void sigmoid_kernel_hb(TensorIterator& iter) {

  TORCH_CHECK(false, "sigmoid_kernel_hb not implemented");
  // TODO: implement the host code for sigmoid kernel
  //       you may refer to add_kernel_hb in
  //       aten/src/ATen/native/hammerblade/AddSub.cpp
  //       as an example
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(sigmoid_stub, &sigmoid_kernel_hb);

}} // namespace at::native
