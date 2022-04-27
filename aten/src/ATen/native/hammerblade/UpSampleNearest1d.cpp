#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/hammerblade/OffloadDef.h>
#include <ATen/native/hammerblade/OffloadUtils.h>

namespace at { namespace native {


namespace {


static void upsample_nearest1d_kernel_hb(TensorIterator& iter) {
  AT_DISPATCH_FLOAT_TYPE_ONLY(iter.dtype(), "upsample_nearest1d_hb", [&]() {
      offload_op_unary(iter, "tensorlib_upsample_nearest1d");
      });
}

} // anonymous namespace

REGISTER_HAMMERBLADE_DISPATCH(upsample_nearest1d_stub, &upsample_nearest1d_hb);

}} // namespace at::native