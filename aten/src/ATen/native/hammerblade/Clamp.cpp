#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor& _clamp__hb(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  return clamp_out(self, self, min, max);
}

Tensor& _clamp_out_hb(
    Tensor& result,
    const Tensor& self,
    optional<Scalar> min,
    optional<Scalar> max) {
  if (min && max) {
    checkBackend("clamp", result, Backend::HammerBlade);
    auto iter = TensorIterator::unary_op(result, self,
        /*check_mem_overlap=*/true);
    clamp_stub(iter.device_type(), iter, *min, *max);
  } else if (max) {
    clamp_max_out(result, self, *max);
  } else if (min) {
    clamp_min_out(result, self, *min);
  } else {
    AT_ERROR("At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor& _clamp_max__hb(Tensor& self, Scalar max) {
  return clamp_max_out(self, self, max);
}

Tensor& _clamp_max_out_hb(Tensor& result, const Tensor& self, Scalar max) {
  checkBackend("clamp_max", result, Backend::HammerBlade);
  auto iter = TensorIterator::unary_op(result, self,
      /*check_mem_overlap=*/true);
  clamp_max_stub(iter.device_type(), iter, max);
  return result;
}

Tensor& _clamp_min__hb(Tensor& self, Scalar min) {
  return clamp_min_out(self, self, min);
}

Tensor& _clamp_min_out_hb(Tensor& result, const Tensor& self, Scalar min) {
  checkBackend("clamp_min", result, Backend::HammerBlade);
  auto iter = TensorIterator::unary_op(result, self,
      /*check_mem_overlap=*/true);
  clamp_min_stub(iter.device_type(), iter, min);
  return result;
}

namespace {

static void clamp_kernel_hb(TensorIterator& iter, Scalar min_scalar, Scalar max_scalar) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "clamp_hb", [&]() {
      auto min = min_scalar.to<scalar_t>();
      auto max = max_scalar.to<scalar_t>();
      offload_op_unary(iter, min, max, "tensorlib_clamp");
      });
}

static void clamp_min_kernel_hb(TensorIterator& iter, Scalar min_scalar) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "clamp_min_hb", [&]() {
      auto min = min_scalar.to<scalar_t>();
      offload_op_unary(iter, min, "tensorlib_clamp_min");
      });
}

static void clamp_max_kernel_hb(TensorIterator& iter, Scalar max_scalar) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "clamp_max_hb", [&]() {
      auto max = max_scalar.to<scalar_t>();
      offload_op_unary(iter, max, "tensorlib_clamp_max");
      });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(clamp_stub, &clamp_kernel_hb);
REGISTER_HAMMERBLADE_DISPATCH(clamp_min_stub, &clamp_min_kernel_hb);
REGISTER_HAMMERBLADE_DISPATCH(clamp_max_stub, &clamp_max_kernel_hb);

}} // namespace at::native
