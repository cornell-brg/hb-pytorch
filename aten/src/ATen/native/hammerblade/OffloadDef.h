#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/hammerblade/HammerBladeContext.h>

namespace at {
namespace native {


//====================================================
// HammerBlade Tensor Struct
//
// This struct defines the tensor layout on HB device.
// HB kernel offloading routines cast PyTorch's tensors
// to this format before loading and launching a kernel.
// The layout of this struct matches that of the C struct
// defined in HB device runtime.
//====================================================

typedef struct {
  uint32_t N;    // Number of elements in the tensor
  uint32_t dims; // Number of dimensions
  eva_t strides; // Pointer to stride vector; number of strides = dims
  eva_t sizes;   // Pointer to sizes vector; number of sizes = dims
  eva_t data;    // Pointer to raw data
#ifdef HB_ENABLE_KERNEL_LOG
  float* storage_head;
  uint32_t storage_numel;
#endif
} hb_mc_tensor_t;

//====================================================
// HammerBlade Vector
//====================================================

typedef struct {
  uint32_t N;
  eva_t data;
} hb_mc_vector_t;

} // namespace native
} // namespace at
