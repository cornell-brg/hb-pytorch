#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native { namespace {

static void sum_kernel_impl(TensorIterator& iter) {

}

}  // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(sum_stub, &sum_kernel_impl);

}}  // namespace at::native
