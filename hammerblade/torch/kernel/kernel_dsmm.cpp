//====================================================================
// Dense-sparse matrix product
// 07/14/2020 Andrew Pareles (amp342@cornell.edu)
//====================================================================


#include <kernel_common.hpp>
#include <cmath>

extern "C" {


/** 
 *  Dense-(sparse.T) matrix multiply.
*/
  __attribute__ ((noinline))  int tensorlib_dstmm(
          hb_tensor_t* out_p, //destination
          hb_tensor_t* a_p, //dense
          hb_tensor_t* b_csc_p,
          hb_tensor_t* b_rows_p,
          hb_tensor_t* b_values_p
          ) { 
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    auto res = HBTensor<float>(out_p);
    auto a = HBTensor<float>(a_p);
    auto b_csc = HBTensor<int>(b_csc_p);
    auto b_rows = HBTensor<int>(b_rows_p);
    auto b_vals = HBTensor<float>(b_values_p);

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

    printf("b_csc:"); 
    for (int i = 0;i< b_csc.numel(); i++){
      printf("%i, ", b_csc(i));
    } printf("\n");

    // for each row of a, do the multiplication
    //(since this is a very even way to distribute the work)
    float sum;
    hb_tiled_for(a_nrows, [&](int a_row) {
      //for each interval [n,n+1] of b_csc (ie each col of b):
      for (int b_col = 0; b_col < b_ncols; b_col++){
        sum = 0;
        //for each nonzero entry in this col of b
        for (int b_row_idx = b_csc(b_col); b_row_idx < b_csc(b_col+1); b_row_idx++){
          int b_row = b_rows(b_row_idx);
          float b_val = b_vals(b_row_idx);
          printf("arow %i, ", a_row);
          printf("bcol %i, ", b_col);
          printf("brow %i, ", b_row);
          printf("bval %f\n", b_val);
          sum += b_val * a(a_row, b_row); //a_col = b_row in dot prod
        }
        printf("\n");
        res(a_row, b_col) = sum;
      }
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_dstmm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
