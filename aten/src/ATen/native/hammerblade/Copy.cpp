#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

static bool copy_requires_temporaries(TensorIterator& iter) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  if (dst_device == src_device) {
    // We never require temporaries for copies on HammerBlade
    // XXX: Be careful with this ...
    TORCH_INTERNAL_ASSERT(dst_device.is_hammerblade(), src_device.is_hammerblade());
    return false;
  }

  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // Contiguous same-dtype copies can always use memcpy
    return false;
  }
  return true;
}

static void copy_hb_to_hb(TensorIterator& iter, bool non_blocking) {
  offload_memcpy(iter);
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

  if (copy_requires_temporaries(iter)) {
    // XXX: this involves recursive calls to copy. Be careful that those copies
    // don't require temporaries or you will cause an infinite recursion!
    auto& dst = iter.tensor(0);
    Tensor dst_contig;
    Tensor src_contig;

    // Type conversions are performed on the CPU for CPU-HB copies and on
    // the src device for other copies.
    // XXX: currently we do *NOT* support copying between different types
    if (iter.device_type(0) == kHAMMERBLADE) {
      dst_contig = dst.is_contiguous() ? dst : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();
    } else {
      bool same_type = iter.dtype(0) == iter.dtype(1);
      dst_contig = (dst.is_contiguous() && same_type) ? dst : at::empty_like(dst, iter.dtype(1), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).expand_as(dst).contiguous();
    }

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return;
  }

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
