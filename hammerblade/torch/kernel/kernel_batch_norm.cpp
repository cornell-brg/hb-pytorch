//====================================================================
// BatchNorm kernel
// 07/11/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_batch_norm_transform_input(
      hb_tensor_t* output,
      hb_tensor_t* input,
      hb_tensor_t* weight,
      hb_tensor_t* bias,
      hb_tensor_t* save_mean,
      hb_tensor_t* save_invstd,
      hb_tensor_t* running_mean,
      hb_tensor_t* running_var,
      bool train,
      double eps) {
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_batch_norm_transform_input,
      hb_tensor_t*,
      hb_tensor_t*,
      hb_tensor_t*,
      hb_tensor_t*,
      hb_tensor_t*,
      hb_tensor_t*,
      hb_tensor_t*,
      hb_tensor_t*,
      bool,
      double);

}
