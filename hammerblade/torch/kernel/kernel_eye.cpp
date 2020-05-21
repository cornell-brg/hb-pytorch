#include <kernel_common.hpp>


extern "C" {

  __attribute__ ((noinline)) int tensorlib_eye(
    hb_tensor_t* output, uint32_t* n, uint32_t* m) {
    HBTensor<float> y(output);
    uint32_t N = *n;
    uint32_t M = *m;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    hb_parallel_for(N, [&](size_t i) {
      for(auto j = 0; j < M; j++) {
        if(i == j) {
          y(i,j) = 1;
        }
        else {
          y(i,j) = 0;
        }
      }
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_eye, hb_tensor_t*, uint32_t*, uint32_t*);

}
