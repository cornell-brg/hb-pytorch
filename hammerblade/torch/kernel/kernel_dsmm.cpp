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
          hb_tensor_t* b_cols_p,
          hb_tensor_t* b_rows_p,
          hb_tensor_t* b_values_p
          ) { 
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    auto res = HBTensor<float>(out_p);
    auto a = HBTensor<float>(a_p);
    auto b_cols = HBTensor<int>(b_cols_p);
    auto b_rows = HBTensor<int>(b_rows_p);
    auto b_vals = HBTensor<float>(b_values_p);
    
    auto a_ncols = a.dim(1);
    auto a_nrows = res.dim(0);
    auto b_ncols = res.dim(1);
    auto nnz = b_vals.numel();

    printf("bvals:"); 
    for (int i = 0;i< b_vals.numel(); i++){
      printf("%f, ", b_vals(i));
    } printf("\n");

    printf("brows:"); 
    for (int i = 0;i< b_rows.numel(); i++){
      printf("%i, ", b_rows(i));
    } printf("\n");

    printf("bcols:"); 
    for (int i = 0;i< b_cols.numel(); i++){
      printf("%i, ", b_cols(i));
    } printf("\n");

    // for each row of res, do the multiplication
    //(since this is a very even way to distribute the work)
    float sum;
    hb_tiled_for(nnz, [&](size_t i) {
      int b_row = b_rows(i);
      int b_col = b_cols(i);
      float b_val = b_vals(i);
      for (int a_row = 0; a_row < a_nrows; a_row++){
        res(a_row, b_col) += b_val * a(a_row, b_row);
      }
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_dsmm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
