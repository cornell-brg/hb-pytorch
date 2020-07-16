//====================================================================
// Dense-sparse matrix product
// 07/14/2020 Andrew Pareles (amp342@cornell.edu)
//====================================================================


#include <kernel_common.hpp>
#include <cmath>

extern "C" {


/** 
 *  Dense-sparse matrix multiply.
*/
  __attribute__ ((noinline))  int tensorlib_dsmm(
          hb_tensor_t* out_p, //destination
          hb_tensor_t* a_p, //dense
          hb_tensor_t* bcsc_p, //sparse
          hb_tensor_t* b_rows_p,
          hb_tensor_t* b_vals_p,
          size_t* dot_prod_len, // i.e. a.size(0) or b.size(1)
          size_t* b_nnz
          ) { 
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    auto a = HBTensor<float>(a_p);
    auto bcsc = HBTensor<float>(bcsc_p);
    auto b_rows = HBTensor<int>(b_rows_p);
    auto b_vals = HBTensor<int>(b_vals_p);
    auto res = HBTensor<float>(out_p);
    auto dp_len = *dot_prod_len;
    auto numel = *b_nnz;

    // for each nonzero element in the sparse matrix
    float sum;
    hb_tiled_for(numel, [&](size_t i) {
      int row = b_rows(i);
      int col = csc(i);
      sum = 0;
      for (int dot = 0; dot < dp_len; dot++){
        sum += a(row, dot) * bcsc(dot, col);
      }
      res(row, col) += sum;
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_dsmm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, size_t*, size_t*)

}
