#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

/* These are FAKE copy kernels
 * They are just using memcpy
 */

static void copy_hb_to_hb(TensorIterator& iter, bool non_blocking) {
  AT_ERROR("On device memcpy not allowed for HammerBlade");
}

static void copy_cpu_to_hb(TensorIterator& iter, bool non_blocking) {
  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  // hb_mc_device_memcpy takes void* as dst and src
  uint32_t nbytes = iter.numel() * iter.element_size(0);
  c10::hammerblade::memcpy_host_to_device(dst, src, nbytes);
}

static void copy_hb_to_cpu(TensorIterator& iter, bool non_blocking) {
  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  // hb_mc_device_memcpy takes void* as dst and src
  uint32_t nbytes = iter.numel() * iter.element_size(0);
  c10::hammerblade::memcpy_device_to_host(dst, src, nbytes);
}

static void copy_kernel_hammerblade(TensorIterator& iter, bool non_blocking) {
  AT_ASSERT(iter.ntensors() == 2);

  // Since this copy kernel is FAKE ...
  // In reality we need to be able to copy between tensors with different dtypes
  TORCH_INTERNAL_ASSERT(iter.dtype(0) == iter.dtype(1),
      "So far HammerBlade copy kernel can only handle tensors with same dtype");

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  // Copy on Device
  if (dst_device.is_hammerblade() && src_device.is_hammerblade()) {
    copy_hb_to_hb(iter, non_blocking);
    return;
  }

  // Copy between CPU and Device
  if (dst_device.is_hammerblade() && src_device.is_cpu()) {
    copy_cpu_to_hb(iter, non_blocking);
  } else if (dst_device.is_cpu() && src_device.is_hammerblade()) {
    copy_hb_to_cpu(iter, non_blocking);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in HammerBlade copy_()");
  }
}


REGISTER_HAMMERBLADE_DISPATCH(copy_stub, &copy_kernel_hammerblade);

} // namespace native
} // namespace at
