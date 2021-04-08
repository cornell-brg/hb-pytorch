//====================================================================
// Dense-sparse matrix product
// 04/07/2021 Neil Adit (na469@cornell.edu)
//====================================================================


#include <kernel_common.hpp>
#include <cmath>

extern "C" {


/** 
 *  Dense-sparse matrix multiply.
*/
  __attribute__ ((noinline))  int tensorlib_sp_mul_(
          hb_tensor_t* result_inds_, //destination
          hb_tensor_t* result_vals_, //destination
          hb_tensor_t* inds_, //indices
          hb_tensor_t* vals_a, //values sparse a
          hb_tensor_t* vals_b //values sparse b
          ) { 
    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    auto indices = HBTensor<int>(inds_);

    auto a = HBTensor<float>(vals_a);
    auto b = HBTensor<float>(vals_b);
    auto res_indices = HBTensor<int>(result_inds_);
    auto res_vals = HBTensor<float>(result_vals_);
    
    auto numel = res_vals.numel(); //nnz

    float mul;
    hb_tiled_for(numel, [&](size_t i) {
      int row = indices(i); 
      int col = indices(i + numel);

      mul = a(i)*b(i);

      // update indices & vals
      res_indices(i) = row;
      res_indices(i + numel) = col;
      res_vals(i) = mul;
    });

    //   End profiling
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sp_mul_, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
