#include <ATen/native/Pool.h>
#include <ATen/native/hammerblade/OffloadDef.h>
#include <ATen/native/hammerblade/OffloadUtils.h>

namespace at {
namespace native {

//===================================================
// Helper functions for converting PyTorch Tensors to
// HammerBlade device Tensors
// Bandhav Veluri, Lin Cheng
//===================================================

static void copy_strides_or_sizes(eva_t tensor_strides, const int64_t* strides, uint32_t dims) {

  if(strides == NULL) {
    return;
  }

  // construct a uint32_t local_strides/sizes
  uint32_t *local_strides = (uint32_t*) malloc(dims * sizeof(uint32_t));
  if(!local_strides) {
    AT_ERROR("Failed to allocate space for tmp strides on host");
  }

  // populate local_strides/sizes
  for(int i=0; i<dims; i++) {
    local_strides[i] = safe_downcast<uint32_t, int64_t>(strides[i]);
  }

  // copy strides/sizes
  void* dst = (void *) ((intptr_t) tensor_strides);
  void* src = (void *) ((intptr_t) local_strides);
  c10::hammerblade::memcpy_host_to_device(dst, src, dims * sizeof(uint32_t));
  free(local_strides);

  return;
}

eva_t create_device_tensor(uint32_t N, uint32_t dims,
                                  const int64_t* strides,
                                  const int64_t* sizes,
                                  const void* data,
#ifdef HB_ENABLE_KERNEL_LOG
                                  void* storage_head,
                                  uint32_t storage_numel,
#endif
                                  std::vector<eva_t>& device_ptrs) {

  eva_t tensor, tensor_strides, tensor_sizes, tensor_data;

  // allocate memory for tensor struct
  tensor = c10::hammerblade::device_malloc(sizeof(hb_mc_tensor_t));

  // allocate memory for strides
  tensor_strides = c10::hammerblade::device_malloc(dims * sizeof(uint32_t));
  device_ptrs.push_back(tensor_strides);

  // allocate memory for sizes
  tensor_sizes = c10::hammerblade::device_malloc(dims * sizeof(uint32_t));
  device_ptrs.push_back(tensor_sizes);

  // tensor struct on host
  hb_mc_tensor_t tensor_host = {
    .N = N,
    .dims = dims,
    .strides = tensor_strides,
    .sizes = tensor_sizes,
    .data = (eva_t)((intptr_t)data),
#ifdef HB_ENABLE_KERNEL_LOG
    .storage_head = storage_head,
    .storage_numel = storage_numel,
#endif
  };

  // copy tensor struct
  void* dst = (void *) ((intptr_t) tensor);
  void* src = (void *) ((intptr_t) &tensor_host);
  c10::hammerblade::memcpy_host_to_device(dst, src, sizeof(hb_mc_tensor_t));

  // copy tensor strides
  copy_strides_or_sizes(tensor_strides, strides, dims);
  // copy tensor sizes
  copy_strides_or_sizes(tensor_sizes, sizes, dims);

  return tensor;
}


eva_t create_device_tensor(const Tensor& tensor,
                                  std::vector<eva_t> device_ptrs) {
  if(!tensor.is_contiguous()) {
    AT_WARN("Offloading non contiguous plain tensor. Make sure you are expecting this!");
  }

  uint32_t N = (uint32_t) tensor.numel();
  uint32_t dims = (uint32_t) tensor.dim();
  const int64_t* strides = (const int64_t*) tensor.strides().data();
  const int64_t* sizes = (const int64_t*) tensor.sizes().data();
  const void* data = (const void*) tensor.data_ptr();

  return create_device_tensor(
      N, dims, strides, sizes, data,
#ifdef HB_ENABLE_KERNEL_LOG
      (void*) tensor.storage().data(),
      (uint32_t) tensor.storage().numel(),
#endif
      device_ptrs);
}

//===================================================
// Helper function for converting PyTorch Vectors to
// HammerBlade device Vectors
// Bandhav Veluri
//===================================================

eva_t create_device_vector(IntArrayRef arr_ref, bool input,
                                  std::vector<eva_t> device_ptrs) {
  uint32_t N = arr_ref.size();
  const int64_t* data = arr_ref.data();
  eva_t data_d = c10::hammerblade::device_malloc(N * sizeof(int32_t));
  device_ptrs.push_back(data_d);

  hb_mc_vector_t vec = {
    .N = N,
    .data = data_d,
  };

  eva_t vec_d = c10::hammerblade::device_malloc(sizeof(hb_mc_vector_t));
  void* dst = (void*) ((intptr_t) vec_d);
  void* src = (void*) ((intptr_t) &vec);
  c10::hammerblade::memcpy_host_to_device(dst, src, sizeof(hb_mc_vector_t));

  if(input) {
    int32_t* local_data = (int32_t*) malloc(N * sizeof(int32_t));
    TORCH_CHECK(local_data, "Memory allocation failed on host");

    for(int i = 0; i < N; ++i) {
      local_data[i] = (int32_t) data[i];
    }

    dst = (void*) ((intptr_t) data_d);
    src = (void*) ((intptr_t) local_data);
    c10::hammerblade::memcpy_host_to_device(
        dst, src, N * sizeof(int32_t));
  }

  return vec_d;
}

//===================================================
// Helper function for mem leak prevention during
// offloading
// Lin Cheng
//===================================================

void cleanup_device(std::vector<eva_t> args, std::vector<eva_t> ptrs) {
  for(int i=0; i<ptrs.size(); i++) {
    c10::hammerblade::device_free(ptrs[i]);
  }
  for(int i=0; i<args.size(); i++) {
    c10::hammerblade::device_free(args[i]);
  }
}


} // namespace native
} // namespace at
