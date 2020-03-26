// ======================================================================
// HammerBlade Offloading
//
// Author: Bandhav Veluri, Lin Cheng
// ======================================================================

#include <ATen/native/TensorIterator.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

 /* -------------------------------------------------------------------------
 * Offloading wrapper
 * --------------------------------------------------------------------------*/

// Offloading operations that have tensors and scalars as arguments
void offload_tensor_scalar_impl(std::vector<Tensor> tensors,
                                std::vector<eva_t> device_scalars,
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
    // Create raw-tensor
    eva_t device_arg = create_device_tensor(arg, device_ptrs);
    device_args.push_back(device_arg);
    // NOTE: here we are assuming all strides need to be copied.
  }

  // Add device_scalars to argument lists
  for(int i=0; i<device_scalars.size(); i++) {
    auto alpha = device_scalars[i];
    device_args.push_back(alpha);
  }

  c10::hammerblade::offload_kernel(kernel, device_args);

  // Need to deallocate those args on device
  cleanup_device(device_args, device_ptrs);

}

// Offload routine for device to device transfers
void offload_memcpy(eva_t dest, eva_t src, uint32_t n) {
  std::vector<eva_t> device_args;

  device_args.push_back(dest);
  device_args.push_back(src);
  device_args.push_back(create_device_scalar(n));

  c10::hammerblade::offload_kernel("tensorlib_memcpy", device_args);
}

// Offload routine for device to device transfers when input is an iter
void offload_memcpy(TensorIterator& iter) {
  offload_op_unary(iter, "tensorlib_copy_hb_to_hb");
}
} // namespace native
} // namespace at
