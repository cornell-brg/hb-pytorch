//====================================================================
// Sampled dense-denseT matrix multiply
// 08/17/2020 Andrew Pareles (amp342@cornell.edu)
//====================================================================


#include <kernel_common.hpp>
#include <cmath>

extern "C" {
/** 
 * For each i,j, if the element a_ij is nonzero,
 * find (b@c.T)_ij. The result goes in out_p.
 * Nonzero entries of a are represented by (row, col).
*/
  __attribute__ ((noinline))  int tensorlib_sddtmm(
          hb_tensor_t* result_inds_p, //destination
          hb_tensor_t* result_vals_p, //destination
          hb_tensor_t* inds_p, //indices
          hb_tensor_t* b_p, //dense
          hb_tensor_t* c_p //dense
          ) { 
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    auto indices = HBTensor<int>(inds_p);

    auto b = HBTensor<float>(b_p);
    auto c = HBTensor<float>(c_p);
    auto res_indices = HBTensor<int>(result_inds_p);
    auto res_vals = HBTensor<float>(result_vals_p);
    
    auto dp_len = b.dim(1); // i.e. b.size(1) or c.size(1)
    auto numel = res_vals.numel(); // i.e. cols.size() or rows.size()

    float sum;
    hb_tiled_for(numel, [&](size_t i) {
      int row = indices(i); // ie indices(0, i)
      int col = indices(i + numel); // ie indices(1, i)
      sum = 0;
      for (int dot = 0; dot < dp_len; dot++){
        sum += b(row, dot) * c(col, dot);
      }
      // update indices & vals
      res_indices(i) = row;
      res_indices(i + numel) = col;
      res_vals(i) = sum;
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sddtmm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
