//==========================================================================================
//The kernel code of changing the data layout of dense vector from HB with SpMV Xcel back to CPU
//11/07/2020 Zhongyuan Zhao(zz546@cornell.edu)
//==========================================================================================
#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_xceltocpu_vector(
    hb_tensor_t* _result,
    hb_tensor_t* _xcel_out) {
    return 0;
  }
  HB_EMUL_REG_KERNEL(tensorlib_xceltocpu_vector, hb_tensor_t*, hb_tensor_t*)
}
