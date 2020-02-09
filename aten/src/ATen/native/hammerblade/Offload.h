#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

void offload_op_binary(TensorIterator& iter, Scalar alpha, const char* kernel);

} // namespace native
} // namespace at
