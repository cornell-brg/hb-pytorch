//====================================================================
// Product reduction kernel
// 11/14/2021 Aditi Agarwal (aa2224@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <hb_reduction.hpp>
extern "C" {

  __attribute__ ((noinline))  int tensorlib_prod(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          uint32_t* num_reduction_dim_p) {
    auto res = HBTensor<float>(t0_p);
    auto input = HBTensor<float>(t1_p);
    uint32_t dim = *num_reduction_dim_p;
    auto ndim = input.ndim();

    
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();
    float prod;




    // if(dim == 1){ //product of elements in each row
    //     for(auto i = start; i < input.dim(0); i += thread_num){
    //         prod = 1;
    //         for(int32_t j = 0; j<input.dim(1);j++){
    //           prod *= input(j,i);
    //         }
    //         res(0,i) = prod;
    //     }
    //     // res(0,0) = 1.0;
    // }else{ //product of elements in each col
    //     for(auto i = start; i < input.dim(1); i += thread_num){
    //         prod = 1;
    //         for(int32_t j = 0; j<input.dim(0);j++){
    //           prod *= input(i,j);
    //         }
    //         res(0,i) = prod;
    //     }
    //     // bsg_printf("no error");
    //     // res(0,0) = 1.0;
    // }


    
    uint32_t elements_to_collect =
      calc_elements_per_output(res, input, dim);

    auto reduce = [](float& partial_result, float in) {
                    // if(!partial_result) partial_result = 1;
                    partial_result *= in;
                  };

    auto project = [](float result) {
                    return result;
                   };

    
    binary_reduction(res, input, ndim, dim,
          elements_to_collect, reduce, project, Reduction::Prod, 1);

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_prod, hb_tensor_t*, hb_tensor_t*, uint32_t*)

}
