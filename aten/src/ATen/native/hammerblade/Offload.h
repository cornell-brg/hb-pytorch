#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

/**
 * HammerBlade Tensor Struct
 *
 * This struct defines the tensor layout on HB device.
 * HB kernel offloading routines cast PyTorch's tensors
 * to this format before loading and launching a kernel.
 * The layout of this struct matches that of the C struct
 * defined in HB device runtime.
 */
typedef struct {
  uint32_t N;    // Number of elements in the tensor
  uint32_t dims; // Number of dimensions
  eva_t strides;  // Pointer to stride vector; number of strides = dims
  eva_t data;    // Pointer to raw data
} hb_mc_tensor_t;

void offload_op_binary(TensorIterator& iter, Scalar alpha, const char* kernel);

void offload_memcpy(eva_t dest, eva_t src, uint32_t n);


} // namespace native
} // namespace at
