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
  constexpr int ntensors = 3; // it must be 3 ... as the assert says above ...

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  at::detail::Array<ScalarType, ntensors> dtypes;
  for (int i = 0; i < ntensors; i++) {
    dtypes[i] = iter.tensor(i).scalar_type();
  }

  int64_t numel = iter.numel();

  AT_WARN("more work to be done");
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
