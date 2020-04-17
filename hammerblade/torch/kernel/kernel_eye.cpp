#include <kernel_common.hpp>


extern "C" {

  __attribute__ ((noinline)) int tensorlib_eye(
    hb_tensor_t* output, long* n, long* m) {
    auto y = HBTensor<float>(output);
    auto N = *n;
    auto M = *m;
    // Start profiling

    bsg_cuda_print_stat_kernel_start();
    for(long i = 0; i < N; i++) {
      for(long j = 0; j < M; j++) {
        if(i == j) {
          y(i,j) = 1;
        }
        else {
          y(i,j) = 0;
        }
      }
    }
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_eye, hb_tensor_t*, long*, long*);

}