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
    // Create raw-tensor
    eva_t device_arg = create_device_tensor(arg, device_ptrs);
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

// Offload routine convolution forward pass
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
  device_args.push_back(create_device_tensor(output, device_ptrs));
  device_args.push_back(create_device_tensor(input, device_ptrs));
  device_args.push_back(create_device_tensor(weight, device_ptrs));
  device_args.push_back(create_device_vector(padding, true, device_ptrs));
  device_args.push_back(create_device_vector(stride, true, device_ptrs));

  c10::hammerblade::offload_kernel(
      "tensorlib_convolution_forward", device_args);
  cleanup_device(device_args, device_ptrs);
}

// Offload routine for covolution bias addition
void offload_convolution_add_bias(const Tensor& output, const Tensor& bias) {
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
  device_args.push_back(create_device_tensor(output, device_ptrs));
  device_args.push_back(create_device_tensor(bias, device_ptrs));

  c10::hammerblade::offload_kernel(
      "tensorlib_convolution_add_bias", device_args);
  cleanup_device(device_args, device_ptrs);
}

} // namespace native
} // namespace at
