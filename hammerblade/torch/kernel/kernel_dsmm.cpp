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
          hb_tensor_t* b_csc_p, //csc of sparse (b = sparse)
          hb_tensor_t* b_rows_p,
          hb_tensor_t* b_values_p,
          size_t* b_nnz,
          ) { 
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    auto res = HBTensor<float>(out_p);
    auto a = HBTensor<float>(a_p);
    auto b_csc = HBTensor<float>(b_csc_p);
    auto b_rows = HBTensor<int>(b_rows_p);
    auto b_vals = HBTensor<int>(b_values_p);

    auto a_nrows = res.dim(0);
    auto b_ncols = res.dim(1);
    auto nnz = b_vals.numel();

    // for each row of res, do the multiplication
    //(since this is a very even way to distribute the work)
    float sum;
    hb_tiled_for(a_nrows, [&](size_t a_row) {
      //for each interval [n,n+1] of b_csc (ie each col of b):
      for (int b_col = 0; b_col < b_ncols; b_col++){
        sum = 0;
        //for each nonzero entry in this col of b
        for (int b_row_idx = b_csc(n); b_row_idx < b_csc(n+1); b_row_idx++){
          int b_row = b_rows(b_row_idx);
          int b_val = b_vals(b_row_idx);
          sum += b_val * a(a_row, b_row); //a_col = b_row in dot prod
        }
        res(a_row, b_col) = sum;
      }
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_dsmm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, size_t*, size_t*)

}
