//====================================================================
// Sampled dense-dense matrix multiply
// 06/29/2020 Andrew Pareles (amp342@cornell.edu)
//====================================================================


#include <kernel_common.hpp>
#include <cmath>

extern "C" {


/** 
 * For each i,j, if the element a_ij is nonzero,
 * find (b@c)_ij. The result goes in out_p.
 * Nonzero entries of a are represented by (col, row).
*/
  __attribute__ ((noinline))  int tensorlib_sddmm(
          hb_tensor_t* out_p, //destination
          hb_tensor_t* col_p, //cols
          hb_tensor_t* row_p, //rows
          hb_tensor_t* b_p, //dense
          hb_tensor_t* c_p, //dense
          size_t* dot_prod_len, // i.e. b.size(0) or c.size(1)
          size_t* nnz
          ) { 
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    auto cols = HBTensor<long>(col_p);
    auto rows = HBTensor<long>(row_p);
    auto b = HBTensor<float>(b_p);
    auto c = HBTensor<float>(c_p);
    auto res = HBTensor<float>(out_p);
    auto dp_len = *dot_prod_len;
    auto numel = *nnz;

    float sum;
    hb_tiled_for(numel, [&](size_t i) {
      long row = rows(i);
      long col = cols(i);
      sum = 0;
      for (int dot = 0; dot < dp_len; dot++){
        sum += b(row, dot) * c(dot, col);
      }
      res(row, col) = sum;
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sddmm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, size_t*, size_t*)

}
