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
int tensorlib_hardtanh(hb_tensor_t* out, hb_tensor_t* t0_p,
                       float* min, float* max)
{
  bsg_printf("Running tensorlib_hardtanh\n");
  bsg_cuda_print_stat_kernel_end();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib_hardtanh, hb_tensor_t*, 
                   hb_tensor_t*, float*, float*)

} /* extern C */
