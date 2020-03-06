#include <cmath>
#include <iostream>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {
namespace {

void add_kernel_hb(TensorIterator& iter, Scalar alpha_scalar) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "add_hb/sub_hb", [&]() {
      offload_op_binary(iter, alpha_scalar, "tensorlib_add");
      });
}

void sub_kernel_hb(TensorIterator& iter, Scalar alpha_scalar) {
  add_kernel_hb(iter, -alpha_scalar);
}

void mul_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "mul_hb", [&]() {
      offload_op_binary(iter, 1.0, "tensorlib_mul");
      });
}

void div_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "div_hb", [&]() {
      offload_op_binary(iter, 1.0, "tensorlib_div");
      });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(add_stub, &add_kernel_hb);
REGISTER_HAMMERBLADE_DISPATCH(sub_stub, &sub_kernel_hb);
REGISTER_HAMMERBLADE_DISPATCH(mul_stub, &mul_kernel_hb);
REGISTER_HAMMERBLADE_DISPATCH(div_stub, &div_kernel_hb);

}} // namespace at::native
