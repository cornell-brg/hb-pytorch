//====================================================================
// Cdist kernel
// 02/17/2021 Neil Adit (na469)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_cdist(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p) {

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    auto res = HBTensor<float>(t0_p);
    auto x1 = HBTensor<float>(t1_p);
    auto x2 = HBTensor<float>(t2_p);

    
    auto ma = res.dim(0);
    auto mb = res.dim(1);
    auto sample_size = x1.dim(1);

    // for each row of a, do the multiplication
    //(since this is a very even way to distribute the work)
    float sum;
    hb_tiled_for(ma, [&](int row) {
      hb_tiled_for(mb, [&](int col) {
        sum = 0;
        for (int el = 0; el < sample_size; el++){
            float temp = x1(row,el) - x2(col,el);
            temp *= temp;
            sum+=temp;
        }
        res(row, col) = sqrt(sum);
      });
    });

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_cdist, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}