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
 * Returns a dense matrix.
*/
  __attribute__ ((noinline))  int tensorlib_sddtmmd(
          hb_tensor_t* result_p, //destination
          hb_tensor_t* row_inds_p, //row indices
          hb_tensor_t* col_inds_p, //col indices
          hb_tensor_t* b_p, //dense
          hb_tensor_t* c_p //dense
          ) { 
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    auto row_indices = HBTensor<int>(row_inds_p);
    auto col_indices = HBTensor<int>(col_inds_p);

    auto b = HBTensor<float>(b_p);
    auto c = HBTensor<float>(c_p);
    auto res = HBTensor<float>(result_p);
    
    auto dp_len = b.dim(1); // i.e. b.size(1) or c.size(1)
    auto numel = row_indices.numel(); // i.e. cols.size() or rows.size()

    float sum;
    hb_tiled_for(numel, [&](size_t i) {
      int row = row_indices(i); // ie indices(0, i)
      int col = col_indices(i); // ie indices(1, i)
      sum = 0;
      for (int dot = 0; dot < dp_len; dot++){
        sum += b(row, dot) * c(col, dot);
      }
      // update res
      res(row, col) = sum;
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sddtmmd, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
