//====================================================================
// Element-wise product kernel
// 11/14/2021 Aditi Agarwal (aa2224@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_prod(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          int dim) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);

    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();
    float prod;
    if(dim){ //product of elements in each row
        for(int32_t i = start; i < input.dim(0); i += thread_num){
            prod = 1;
            for(int32_t j = 0; j<input.dim(1);j++)
                prod *= input(i,j);
            res(0,i) = prod;
        }
    }else{ //product of elements in each col
        for(int32_t i = start; i < input.dim(1); i += thread_num){
            prod = 1;
            for(int32_t j = 0; j<input.dim(1);j++)
                prod *= input(j,i);
            res(0,i) = prod;
        }
    }
    

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_prod, hb_tensor_t*, hb_tensor_t*, int)

}