#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

 /* ------------------------------------------------------------------------------------
 * Helper functions
 * Adapted from bespoke-silicon-group/pytorch branch hb_pytorch
 * Author: Bandhav Veluri, Lin Cheng
 * -------------------------------------------------------------------------------------*/

static hb_mc_eva_t create_device_tensor(uint32_t N, uint32_t dims, const int64_t* strides,
                                  const void* data, bool copy, std::vector<eva_t>& device_ptrs) {

  hb_mc_eva_t tensor, tensor_strides, tensor_data;

  // allocate memory for tensor struct
  tensor = c10::hammerblade::device_malloc(sizeof(hb_mc_tensor_t));

  // allocate memory for strides
  tensor_strides = c10::hammerblade::device_malloc(dims * sizeof(uint32_t));
  device_ptrs.push_back(tensor_strides);

  // tensor struct on host
  hb_mc_tensor_t tensor_host = {
    .N = N,
    .dims = dims,
    .strides = tensor_strides,
    .data = (hb_mc_eva_t)((intptr_t)data),
  };

  // copy tensor struct
  void* dst = (void *) ((intptr_t) tensor);
  void* src = (void *) ((intptr_t) &tensor_host);
  c10::hammerblade::memcpy_host_to_device(dst, src, sizeof(hb_mc_tensor_t));

  if(copy) {
    // copy strides
    dst = (void *) ((intptr_t) tensor_strides);
    src = (void *) ((intptr_t) strides);
    c10::hammerblade::memcpy_host_to_device(dst, src, N * sizeof(float));
  }

  return tensor;
}


static hb_mc_eva_t create_device_scalar(float alpha) {
  hb_mc_eva_t alpha_d;

  alpha_d = c10::hammerblade::device_malloc(sizeof(float));

  void* src = (void*) ((intptr_t) &alpha);
  void* dst = (void*) ((intptr_t) alpha_d);
  c10::hammerblade::memcpy_host_to_device(dst, src, sizeof(float));

  return alpha_d;
}

static void offload_kernel(const char* kernel, std::vector<eva_t> args) {

  uint32_t* cuda_argv = (uint32_t*) malloc(args.size() * sizeof(eva_t));
  if(!cuda_argv) {
    AT_ERROR("Falied to allocate cuda_argv!");
  }

  for(int i=0; i<args.size(); ++i) {
    cuda_argv[i] = args[i];
  }

  AT_WARN("TODO: actually invoke a HB kernel");

  free(cuda_argv);
}

static void cleanup_device(std::vector<eva_t> args, std::vector<eva_t> ptrs) {
  for(int i=0; i<ptrs.size(); i++) {
    c10::hammerblade::device_free(ptrs[i]);
  }
  for(int i=0; i<args.size(); i++) {
    c10::hammerblade::device_free(args[i]);
  }
}
 /* ------------------------------------------------------------------------------------
 * Offloading wrapper
 * -------------------------------------------------------------------------------------*/

void offload_op_binary_impl(TensorIterator& iter, Scalar alpha, const char* kernel) {
  kernel += strlen("tensorlib_"); // remove prepending tag in the kernel name

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3); // output, input1, and input2

  // It is very important to use serial_for_each here, since we assume a single
  // HammerBlade device in the system
  iter.serial_for_each([&](char** data, const int64_t* strides, int64_t n) {
    // Device pointers to tensors on the device
    std::vector<hb_mc_eva_t> device_args;
    std::vector<hb_mc_eva_t> device_ptrs;

    // Allocate device tensors and copy the data
    for(int i=0; i<iter.ntensors(); i++) {
      // Iterate over all tensors: a, b and result, to create
      // corresponding tensors on the device.
      hb_mc_eva_t device_arg = create_device_tensor(n, iter.ndim(),
          &strides[i], data[i], i!=0, device_ptrs);
      device_args.push_back(device_arg);
    }
    device_args.push_back(create_device_scalar(alpha.to<float>()));

    offload_kernel(kernel, device_args);

    // Need to deallocate those args on device
    cleanup_device(device_args, device_ptrs);

  }, {0, iter.numel()});

  iter.cast_outputs();
}

void offload_op_binary(TensorIterator& iter, Scalar alpha, const char* kernel) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3); // output, input1, and input2

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).is_hammerblade());
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      offload_op_binary(sub_iter, alpha, kernel);
    }
    return;
  }

  offload_op_binary_impl(iter, alpha, kernel);
}

} // namespace native
} // namespace at
