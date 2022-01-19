//====================================================================
// Max reduction kernel
// 11/14/2021 Aditi Agarwal (aa2224@cornell.edu)
// 01/11/2022 Zhongyuan Zhao (zz546@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <hb_reduction.hpp>
extern "C" {

  __attribute__ ((noinline))  int tensorlib_max_dim(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          uint32_t* num_reduction_dim_p) {
    auto res = HBTensor<float>(t0_p);
    auto indices = HBTensor<int32_t>(t1_p);
    auto input = HBTensor<float>(t2_p);
    uint32_t dim = *num_reduction_dim_p;
    auto ndim = input.ndim();

    
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();
    float prod;
    
    uint32_t elements_to_collect =
      calc_elements_per_output(res, input, dim);

    auto reduce = [](float& curr_max, float in) {
                   if(in>curr_max) {
                      curr_max = in;
                   }
                  };
    auto reduce1 = [](float& curr_max, int32_t& curr_idx, int32_t idx, float in) {
                    if(in>curr_max) {
                      curr_max = in;
                      curr_idx = idx;
                    }
                   };

    auto project = [](float result) {
                    return result;
                   };

    auto project_int = [](int32_t result) {
                    return result;
                   };

    
    hybrid_reduction(res, indices, input, ndim, dim,
          elements_to_collect, reduce, project, reduce1, project_int);

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_max_dim, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, uint32_t*)

}
