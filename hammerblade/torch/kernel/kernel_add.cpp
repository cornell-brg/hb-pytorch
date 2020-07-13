//====================================================================
// Element-wise add kernel
// 03/05/2020 Lin Cheng and Bandhav Veluri (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {
// float version
  __attribute__ ((noinline))  int tensorlib_add(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          float* alpha_p) {
    auto c = HBTensor<float>(t0_p);
    auto a = HBTensor<float>(t1_p);
    auto b = HBTensor<float>(t2_p);
    float alpha = *alpha_p;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(c, a, b,
      [&](float a, float b) {
        return a + alpha * b;
    });
    
    // End profiling
    bsg_cuda_print_stat_kernel_end();
    // Sync
    g_barrier.sync();
    return 0;
  }
// long long version (64 bit int)
  __attribute__ ((noinline))  int tensorlib_add_Long(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          long long* alpha_p) {
    auto c = HBTensor<long long>(t0_p);
    auto a = HBTensor<long long>(t1_p);
    auto b = HBTensor<long long>(t2_p);
    long long alpha = *alpha_p;

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(c, a, b,
      [&](long long a, long long b) {
	return a + alpha * b;
    });

    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }
// int version (32 bit int)  
    __attribute__ ((noinline))  int tensorlib_add_Int(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          int* alpha_p) {
    auto c = HBTensor<int>(t0_p);
    auto a = HBTensor<int>(t1_p);
    auto b = HBTensor<int>(t2_p);
    int alpha = *alpha_p;
    
    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(c, a, b,
      [&](int a, int b) {
	return a + alpha * b;
    });

    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_add, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*)
  HB_EMUL_REG_KERNEL(tensorlib_add_Int, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, int*)
  HB_EMUL_REG_KERNEL(tensorlib_add_Long, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, long long*)

}
