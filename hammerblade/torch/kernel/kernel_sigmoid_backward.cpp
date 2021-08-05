//====================================================================
// Sigmoid backward kernel
// 08/05/2021 Niklas Schmelzle (jms854@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_sigmoid_backward(
          hb_tensor_t* grad_output,
          hb_tensor_t* output,
          hb_tensor_t* grad_input) {

    auto input_a = HBTensor<float>(grad_output);
    auto input_b = HBTensor<float>(output);
    auto result = HBTensor<float>(grad_input);

    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    hb_tiled_foreach(
      [](float a, float b){
        // grad_input = result = a*(1-b)*b
        a = a * b;
        b = 1 - b;
        a = a * b;
        return a;
      },
      input_a, input_b, result);

    // End profiling
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sigmoid_backward, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}

