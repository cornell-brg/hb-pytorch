//====================================================================
// tanh kernel
//====================================================================
// Returns a new tensor with the tangent of the elements of input
// Used in RNN.
//
// Authors : Jack Weber
// Date    : 05/07/2020

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

//====================================================================
// tensorlib_tanh
//====================================================================
// This is the tanh kernel for tensors with float elements.

__attribute__ ((noinline))
int tensorlib_tanh( hb_tensor_t* t0_p, hb_tensor_t* t1_p)
{
  auto res = HBTensor<float>(t0_p);
  auto input = HBTensor<float>(t1_p);

  bsg_cuda_print_stat_kernel_start();
  hb_tiled_foreach(
    [](float a) {
      return tanh(a);
  },
  res, input);

  bsg_cuda_print_stat_kernel_end();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib_tanh, hb_tensor_t*, hb_tensor_t*)

} /* extern C */
