#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

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

    // Allocate device tensors and copy the data
    for(int i=0; i<iter.ntensors(); i++) {
      // Iterate over all tensors: a, b and result, to create
      // corresponding tensors on the device.
      hb_mc_eva_t device_arg = create_device_tensor(n, iter.ndim(),
          &strides[i], data[i]);
      device_args.push_back(device_arg);
    }
    device_args.push_back(create_device_scalar(alpha.to<float>()));

    offload_kernel(kernel, device_args);
  });

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
