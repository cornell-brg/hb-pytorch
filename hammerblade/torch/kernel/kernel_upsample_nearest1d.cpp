//====================================================================
// Upsample Nearest 1d kernel
// 4/26/2022 Aditi Agarwal (aa2224@cornell.edu)
//====================================================================


#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_upsample_nearest1d(
          hb_tensor_t* t0_p,
          int32_t output_size) {
    auto input = HBTensor<float>(t0_p);
    
    int32_t scale_factor = output_size / input.dim(0); // not sure if this is right

    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    size_t end = output_size;
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();
    float res[output_size];

    for (int32_t i = start; i < end; i = i + thread_num) {
        for (int32_t j = 0; j < scale_factor; j++){
            res[i*scale_factor + j] = input(i);
        }
    }

    for (int32_t i = 0; i < output_size; i++) {
        input(i) = res[i];
    }
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_upsample_nearest1d, int32_t)

}
