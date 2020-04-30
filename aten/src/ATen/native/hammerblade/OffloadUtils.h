#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>

#include <ATen/native/hammerblade/OffloadDef.h>

namespace at {
namespace native {

//============================================
// Helper functions
// Bandhav Veluri, Lin Cheng
//============================================


eva_t create_device_tensor(uint32_t N, uint32_t dims,
                                  const int64_t* strides,
                                  const int64_t* sizes,
                                  const void* data,
#ifdef HB_ENABLE_KERNEL_LOG
                                  float* storage_head,
                                  uint32_t storage_numel,
#endif
                                  std::vector<eva_t>& device_ptrs);

eva_t create_device_tensor(const Tensor& tensor,
                                  std::vector<eva_t> device_ptrs);

eva_t create_device_vector(IntArrayRef arr_ref, bool input,
                                  std::vector<eva_t> device_ptrs);

void cleanup_device(std::vector<eva_t> args, std::vector<eva_t> ptrs);

//===================================================
// Helper function for unwrapping PyTorch Scalars
// Bandhav Veluri, Lin Cheng
//===================================================

template<typename T>
inline eva_t create_device_scalar(T alpha) {
  eva_t alpha_d;

  alpha_d = c10::hammerblade::device_malloc(sizeof(T));

  void* src = (void*) ((intptr_t) &alpha);
  void* dst = (void*) ((intptr_t) alpha_d);
  c10::hammerblade::memcpy_host_to_device(dst, src, sizeof(T));

  return alpha_d;
}

} // namespace native
} // namespace at
