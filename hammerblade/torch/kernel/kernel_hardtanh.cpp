//====================================================================
// Element-wise hardtanh kernel
// 03/06/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

//====================================================================
// tensorlib_hardtanh
//====================================================================
// This is the tanh kernel for tensors with float elements.

__attribute__ ((noinline))
int tensorlib_hardtanh(hb_tensor_t* t0_p, hb_tensor_t* t1_p,
                       float* min_, float* max_)
{
  auto res = HBTensor<float>(t0_p);
  auto input = HBTensor<float>(t1_p);
  float max = *max_;
  float min = *min_;

  bsg_cuda_print_stat_kernel_start();
  hb_tiled_foreach(
    [min, max](float a) {
      if (a < min)
        return min;
      else if (a > max)
        return max;
      else
        return a;
  },
  res, input);

  bsg_printf("%f\n", res(0,0));

  bsg_cuda_print_stat_kernel_end();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib_hardtanh, hb_tensor_t*, 
                   hb_tensor_t*, float*, float*)

} /* extern C */
