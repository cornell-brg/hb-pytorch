//============================================================================
// Sparse matrix multiply dense matrix (SpMM) kernel using row-wise product
// and CSR compression format for sparse matrix
// 17/02/2021 Zhongyuan Zhao, Yuxiang Long (zz546@cornell.edu)
//============================================================================

#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_spmm(
    hb_tensor_t* _result,
    hb_tensor_t* _csr_hb, //CSR mode
    hb_tensor_t* _indices,
    hb_tensor_t* _values,
    hb_tensor_t* _dense) {
    
    auto result = HBTensor<float>(_result);
    auto csr = HBTensor<int>(_csr_hb);  //CSR mode
    auto indices = HBTensor<int>(_indices);
    auto values = HBTensor<float>(_values);
    auto dense = HBTensor<float>(_dense);
    // result(m, n) = sparse(m, k) * dense (k, n) 
    uint32_t m = result.dim(0);
    uint32_t k = dense.dim(0);
    uint32_t n = dense.dim(1);
    uint32_t nnz = values.numel();
 
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    size_t end = m;
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();
    float tmp_out[OUTPUT_BUF_SIZE];
//    int num_iter = std::ceil((float)OUTPUT_BUF_SIZE / (float)n);   
    for (int32_t i = start; i < end; i = i + thread_num) {
      for(int32_t dense_col = 0; dense_col < n; dense_col = dense_col + OUTPUT_BUF_SIZE) {
        int col_num = (dense_col + OUTPUT_BUF_SIZE) > n ? (n - dense_col) : OUTPUT_BUF_SIZE;
        int col_end = dense_col + col_num;
        for(int idx = 0; idx < col_num; idx++) {
          tmp_out[idx] = 0.0;
        }

        for(uint32_t col_index = csr(i); col_index < csr(i+1); col_index++) { //CSR MODE
          float val = values(col_index);
          int row_num = indices(col_index);
          int j = dense_col;
          for(; j <= col_end - 8; j = j + 8) {
            register float d0 = dense(row_num, j);
            register float d1 = dense(row_num, j+1);
            register float d2 = dense(row_num, j+2);
            register float d3 = dense(row_num, j+3);
            register float d4 = dense(row_num, j+4);
            register float d5 = dense(row_num, j+5);
            register float d6 = dense(row_num, j+6);
            register float d7 = dense(row_num, j+7);
            tmp_out[j % OUTPUT_BUF_SIZE] += val * d0;
            tmp_out[(j+1) % OUTPUT_BUF_SIZE] += val * d1;
            tmp_out[(j+2) % OUTPUT_BUF_SIZE] += val * d2;
            tmp_out[(j+3) % OUTPUT_BUF_SIZE] += val * d3;
            tmp_out[(j+4) % OUTPUT_BUF_SIZE] += val * d4;
            tmp_out[(j+5) % OUTPUT_BUF_SIZE] += val * d5;
            tmp_out[(j+6) % OUTPUT_BUF_SIZE] += val * d6;
            tmp_out[(j+7) % OUTPUT_BUF_SIZE] += val * d7;
          }
          for(; j < col_end; j++){
            register float d = dense(row_num, j);
            tmp_out[j % OUTPUT_BUF_SIZE] += val * d;
          }
        }
        
        int k = dense_col;
        for(; k <= col_end - 8; k = k + 8) {
          result(i, k) = tmp_out[k % OUTPUT_BUF_SIZE];
          result(i, k+1) = tmp_out[(k+1) % OUTPUT_BUF_SIZE];
          result(i, k+2) = tmp_out[(k+2) % OUTPUT_BUF_SIZE];
          result(i, k+3) = tmp_out[(k+3) % OUTPUT_BUF_SIZE];
          result(i, k+4) = tmp_out[(k+4) % OUTPUT_BUF_SIZE];
          result(i, k+5) = tmp_out[(k+5) % OUTPUT_BUF_SIZE];
          result(i, k+6) = tmp_out[(k+6) % OUTPUT_BUF_SIZE];
          result(i, k+7) = tmp_out[(k+7) % OUTPUT_BUF_SIZE];
        }
        for(; k < col_end; k++){
          result(i, k) = tmp_out[k % OUTPUT_BUF_SIZE];
        }    
      }        
    }  

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }  

  HB_EMUL_REG_KERNEL(tensorlib_spmm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
}
