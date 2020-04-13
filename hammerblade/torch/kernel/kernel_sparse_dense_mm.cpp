//============================================================================
// Sparse matrix multiply dense matrix kernel
// 04/05/2020 Zhongyuan Zhao, Michael Rivera (zz546@cornell.edu)
//============================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_sparse_dense_mm(
    bsg_tensor_t* _result,
    bsg_tensor_t* _csr_hb,
    bsg_tensor_t* _indices,
    bsg_tensor_t* _values,
    bsg_tensor_t* _dense) {
    
    auto result = BSGTensor<float>(_result);
    auto csr = BSGTensor<int>(_csr_hb);
    auto indices = BSGTensor<int>(_indices);
    auto values = BSGTensor<float>(_values);
    auto dense = BSGTensor<float>(_dense);
    // result(m, n) = sparse(m, k) * dense (k, n) 
    uint32_t m = result.dim(0);
    uint32_t k = dense.dim(0);
    uint32_t n = dense.dim(1);

    size_t len_per_tile = m  / (bsg_tiles_X * bsg_tiles_Y) + 1;
    size_t start = len_per_tile * __bsg_id;
    size_t end = start + len_per_tile;
    end = (end > m) ? m : end;
    
    
    bsg_cuda_print_stat_kernel_start();
   
    for (uint32_t i = start; i < end; i++) {
      for(uint32_t dense_col = 0; dense_col < n; dense_col++) {
        for (uint32_t col_index = csr(i); col_index < csr(i+1); col_index++) {
         //  uint32_t result_index = i * n + dense_col;
         //  result(result_index) = result(result_index) + values(col_index) * dense(indices(col_index + indices.stride(0)) * n + dense_col);
          result(i, dense_col) = result(i, dense_col) + values(col_index) * dense(indices(1, col_index), dense_col);
        }
      }   
    }  

    bsg_cuda_print_stat_kernel_end();
    return 0;
  }  

  HB_EMUL_REG_KERNEL(tensorlib_sparse_dense_mm, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*)
}
