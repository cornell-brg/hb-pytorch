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


void offload_tensorlist_scalar_impl(TensorList list, std::vector<Tensor> tensors,
                                    std::vector<eva_t> device_scalars, const char* kernel) {

  // Device pointers to tensors on the device
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;

  // Create a buffer on device to hold these list member eva's
  eva_t* host_list   = (eva_t*) malloc(list.size() * sizeof(eva_t));
  eva_t  device_list =  c10::hammerblade::device_malloc(list.size() * sizeof(eva_t));
  std::cout << "device_list at eva " << device_list << std::endl;

  // Handle the TensorList
  for(int i=0; i<list.size(); i++) {
    auto arg = list[i];
    TORCH_INTERNAL_ASSERT(arg.device().is_hammerblade())
    eva_t list_arg = create_device_tensor(arg, device_ptrs);
    std::cout << " created tensor at eva " << list_arg << std::endl;
    host_list[i] = list_arg;
  }

  // copy list device pointers (eva's)
  void* dst = (void *) ((intptr_t) device_list);
  void* src = (void *) ((intptr_t) host_list);
  c10::hammerblade::memcpy_host_to_device(dst, src, list.size() * sizeof(eva_t));
  device_args.push_back(device_list);
  free(host_list);

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
  std::string memcpy_kernel = "tensorlib_copy_";
  memcpy_kernel += c10::toString(iter.dtype(1));
  memcpy_kernel += "_to_";
  memcpy_kernel += c10::toString(iter.dtype(0));
  offload_op_unary(iter, memcpy_kernel.c_str());
}

} // namespace native
} // namespace at
