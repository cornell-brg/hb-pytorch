#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

/* ---------------------------------------------------------------------
* Helper functions
* Adapted from bespoke-silicon-group/pytorch branch hb_pytorch
* Author: Bandhav Veluri, Lin Cheng
* ----------------------------------------------------------------------*/

static eva_t create_device_tensor(uint32_t N, uint32_t dims,
                                  const int64_t* strides,
                                  const void* data,
                                  std::vector<eva_t>& device_ptrs) {

  eva_t tensor, tensor_strides, tensor_data;

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
    .data = (eva_t)((intptr_t)data),
  };

  // copy tensor struct
  void* dst = (void *) ((intptr_t) tensor);
  void* src = (void *) ((intptr_t) &tensor_host);
  c10::hammerblade::memcpy_host_to_device(dst, src, sizeof(hb_mc_tensor_t));

  // construct a uint32_t local_strides
  uint32_t *local_strides = (uint32_t*) malloc(dims * sizeof(uint32_t));
  if(!local_strides) {
    AT_ERROR("Failed to allocate space for tmp strides on host");
  }
  // populate local_strides
  for(int i=0; i<dims; i++) {
    local_strides[i] = (uint32_t)strides[i];
  }
  // copy strides
  dst = (void *) ((intptr_t) tensor_strides);
  src = (void *) ((intptr_t) local_strides);
  c10::hammerblade::memcpy_host_to_device(dst, src, dims * sizeof(uint32_t));
  free(local_strides);

  return tensor;
}

static eva_t create_device_tensor(const Tensor& tensor, bool input,
                                  std::vector<eva_t> device_ptrs) {
  uint32_t N = (uint32_t) tensor.numel();
  uint32_t dims = (uint32_t) tensor.dim();
  const int64_t* strides = (const int64_t*) tensor.strides().data();
  const void* data = (const void*) tensor.data_ptr();

  return create_device_tensor(N, dims, strides, data, input, device_ptrs);
}

static eva_t create_device_vector(IntArrayRef arr_ref, bool input,
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

template<typename T>
static eva_t create_device_scalar(T alpha) {
  eva_t alpha_d;

  alpha_d = c10::hammerblade::device_malloc(sizeof(T));

  void* src = (void*) ((intptr_t) &alpha);
  void* dst = (void*) ((intptr_t) alpha_d);
  c10::hammerblade::memcpy_host_to_device(dst, src, sizeof(T));

  return alpha_d;
}


static void cleanup_device(std::vector<eva_t> args, std::vector<eva_t> ptrs) {
  for(int i=0; i<ptrs.size(); i++) {
    c10::hammerblade::device_free(ptrs[i]);
  }
  for(int i=0; i<args.size(); i++) {
    c10::hammerblade::device_free(args[i]);
  }
}


 /* -------------------------------------------------------------------------
 * Offloading wrapper
 * --------------------------------------------------------------------------*/

//=======================================================================
// Offloading operations that have tensors and scalars as arguments
//=======================================================================

void offload_tensor_scalar_impl(std::vector<Tensor> tensors, std::vector<Scalar> scalars,
                                const char* kernel) {

  // Device pointers to tensors on the device
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;

  // Allocate device tensors and copy the data
  for(int i=0; i<tensors.size(); i++) {
    // Iterate over all tensors to create
    // corresponding tensors on the device.
    auto arg = tensors[i];
    TORCH_INTERNAL_ASSERT(arg.device().is_hammerblade())
    // Read low-level meta-data from argument tensor
    uint64_t n = arg.numel();
    uint32_t dims = arg.dim();
    const int64_t* strides = arg.strides().data();
    TORCH_INTERNAL_ASSERT(arg.strides().size() == dims);
    const void* data = arg.data_ptr();
    // Create raw-tensor
    eva_t device_arg = create_device_tensor(n, dims,
        strides, data, device_ptrs);
    device_args.push_back(device_arg);
    // NOTE: here we are assuming all strides need to be copied.
  }

  // Allocate device scalars and copy the data
  for(int i=0; i<scalars.size(); i++) {
    auto alpha = scalars[i];
    device_args.push_back(create_device_scalar(alpha.to<float>()));
  }

  c10::hammerblade::offload_kernel(kernel, device_args);

  // Need to deallocate those args on device
  cleanup_device(device_args, device_ptrs);

}

//=======================================================================
// Offloading operations that use TensorIterator
//=======================================================================

void offload_iterator_op_impl(TensorIterator& iter, Scalar alpha,
    const char* kernel, uint32_t ntensors) {

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == ntensors);

  // It is very important to use serial_for_each here, since we assume a single
  // HammerBlade device in the system
  iter.serial_for_each([&](char** data, const int64_t* strides, int64_t n) {
    // Device pointers to tensors on the device
    std::vector<eva_t> device_args;
    std::vector<eva_t> device_ptrs;

    // Allocate device tensors and copy the data
    for(int i=0; i<iter.ntensors(); i++) {
      // Iterate over all tensors to create
      // corresponding tensors on the device.
      eva_t device_arg = create_device_tensor(n, iter.ndim(),
          &strides[i], data[i], device_ptrs);
      device_args.push_back(device_arg);
    }
    device_args.push_back(create_device_scalar(alpha.to<float>()));

    c10::hammerblade::offload_kernel(kernel, device_args);

    // Need to deallocate those args on device
    cleanup_device(device_args, device_ptrs);

  }, {0, iter.numel()});

  iter.cast_outputs();
}

#define HB_OFFLOAD_NULLARY_OP(iter, alpha, kernel) offload_iterator_op_impl(iter, alpha, kernel, 1)
#define HB_OFFLOAD_UNARY_OP(iter, alpha, kernel) offload_iterator_op_impl(iter, alpha, kernel, 2)
#define HB_OFFLOAD_BINARY_OP(iter, alpha, kernel) offload_iterator_op_impl(iter, alpha, kernel, 3)

// Overload for the 2 Scalar case

void offload_iterator_op_impl(TensorIterator& iter, Scalar alpha, Scalar beta,
    const char* kernel, uint32_t ntensors) {

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == ntensors);

  // It is very important to use serial_for_each here, since we assume a single
  // HammerBlade device in the system
  iter.serial_for_each([&](char** data, const int64_t* strides, int64_t n) {
    // Device pointers to tensors on the device
    std::vector<eva_t> device_args;
    std::vector<eva_t> device_ptrs;

    // Allocate device tensors and copy the data
    for(int i=0; i<iter.ntensors(); i++) {
      // Iterate over all tensors to create
      // corresponding tensors on the device.
      eva_t device_arg = create_device_tensor(n, iter.ndim(),
          &strides[i], data[i], device_ptrs);
      device_args.push_back(device_arg);
    }
    device_args.push_back(create_device_scalar(beta.to<float>()));
    device_args.push_back(create_device_scalar(alpha.to<float>()));

    c10::hammerblade::offload_kernel(kernel, device_args);

    // Need to deallocate those args on device
    cleanup_device(device_args, device_ptrs);

  }, {0, iter.numel()});

  iter.cast_outputs();
}

#define HB_OFFLOAD_NULLARY_OP_2S(iter, beta, alpha, kernel) offload_iterator_op_impl(iter, beta, alpha, kernel, 1)
#define HB_OFFLOAD_UNARY_OP_2S(iter, beta, alpha, kernel) offload_iterator_op_impl(iter, beta, alpha, kernel, 2)
#define HB_OFFLOAD_BINARY_OP_2S(iter, beta, alpha, kernel) offload_iterator_op_impl(iter, beta, alpha, kernel, 3)

//=======================================================================
// Offload routine for binary operations
//=======================================================================

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

  HB_OFFLOAD_BINARY_OP(iter, alpha, kernel);
}

//=======================================================================
// Offload routine for unary operations
//=======================================================================

void offload_op_unary(TensorIterator& iter, Scalar alpha, const char* kernel) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2); // output and input1

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

  HB_OFFLOAD_UNARY_OP(iter, alpha, kernel);

}

//=======================================================================
// Offload routine for nullary operations
//=======================================================================

void offload_op_nullary(TensorIterator& iter, Scalar alpha, const char* kernel) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 1); // output

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

  HB_OFFLOAD_NULLARY_OP(iter, alpha, kernel);

}

//=======================================================================
// Offload routine for binary operations
//=======================================================================
// Overload for the 2 Scalar case

void offload_op_binary(TensorIterator& iter, Scalar beta, Scalar alpha, const char* kernel) {
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

  HB_OFFLOAD_BINARY_OP_2S(iter, beta, alpha, kernel);
}

//=======================================================================
// Offload routine for unary operations
//=======================================================================
// Overload for the 2 Scalar case

void offload_op_unary(TensorIterator& iter, Scalar beta, Scalar alpha, const char* kernel) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2); // output and input1

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

  HB_OFFLOAD_UNARY_OP_2S(iter, beta, alpha, kernel);

}

//=======================================================================
// Offload routine for nullary operations
//=======================================================================
// Overload for the 2 Scalar case

void offload_op_nullary(TensorIterator& iter, Scalar beta, Scalar alpha, const char* kernel) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 1); // output

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

  HB_OFFLOAD_NULLARY_OP_2S(iter, beta, alpha, kernel);

}

//=======================================================================
// Offload routine for device to device transfers
//=======================================================================

void offload_memcpy(eva_t dest, eva_t src, uint32_t n) {
  std::vector<eva_t> device_args;

  device_args.push_back(dest);
  device_args.push_back(src);
  device_args.push_back(create_device_scalar(n));

  c10::hammerblade::offload_kernel("tensorlib_memcpy", device_args);
}

void offload_convolution_forward(Tensor& output, const Tensor& input,
    const Tensor& weight, IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups) {

  // Dimension check
  TORCH_CHECK(output.dim() == 4, "Only 2d convolution supported now.");

  // Dilation check
  bool dilation_check = true;
  for(auto d : dilation) {
    if(d != 1) {
      TORCH_WARN("dilation[i] = ", d);
      dilation_check = false;
      break;
    }
  }
  TORCH_CHECK(dilation_check,
        "dilation = ", dilation,
        " is not supported by HB yet.",
        " Make sure dilation is all ones.");

  // Groups check
  TORCH_CHECK(groups == 1,
      "Grouped convolution not supported by HB yet."
      " Make sure groups = 1.");

  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
  device_args.push_back(create_device_tensor(output, false, device_ptrs));
  device_args.push_back(create_device_tensor(input, true, device_ptrs));
  device_args.push_back(create_device_tensor(weight, true, device_ptrs));
  device_args.push_back(create_device_vector(padding, true, device_ptrs));
  device_args.push_back(create_device_vector(stride, true, device_ptrs));

  c10::hammerblade::offload_kernel(
      "tensorlib_convolution_forward", device_args);
  cleanup_device(device_args, device_ptrs);
}

} // namespace native
} // namespace at
