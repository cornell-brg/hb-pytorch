//====================================================================
// Element-wise add kernel
// 03/05/2020 Lin Cheng and Bandhav Veluri (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

__attribute__((noinline)) void vec_add(
             __remote float* NOALIAS C,
             __remote float* NOALIAS A,
             __remote float* NOALIAS B,
             float alpha) {
  int start = __bsg_id*1000;
  int end = (__bsg_id + 1) * 1000;

  for(int n = 0; n < 2; ++n)
    if(__bsg_id < 100)
      UNROLL(16) for(int i = start; i < end; ++i) {
        C[i] = A[i] + alpha * B[i];
      }
}

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_add(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          float* alpha_p) {
    auto c = HBTensor<float>(t0_p);
    auto a = HBTensor<float>(t1_p);
    auto b = HBTensor<float>(t2_p);
    float alpha = *alpha_p;

    bsg_cuda_print_stat_kernel_start();

    auto A = (__remote float*) a.data_ptr();
    auto B = (__remote float*) b.data_ptr();
    auto C = (__remote float*) c.data_ptr();
    
    vec_add(C, A, B, alpha);

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_add, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*)

}
