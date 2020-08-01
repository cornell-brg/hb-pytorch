//============================================================================
// Sparse matrix multiply dense matrix kernel
// 04/05/2020 Zhongyuan Zhao, Michael Rivera (zz546@cornell.edu)
//============================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_sparse_dense_mm(
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
    uint32_t v = values.numel();

    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    size_t end = m;
    
    bsg_cuda_print_stat_kernel_start();

    float temp[1];

    for (uint32_t i = start; i < end; i = i + thread_num) {
      for(uint32_t dense_col = 0; dense_col < n; dense_col++) {
        temp[0] = 0.0;
        for(uint32_t col_index = csr(i); col_index < csr(i+1); col_index++) { //CSR MODE
         temp[0] = temp[0] + values(col_index) * dense(indices(col_index), dense_col); //CSR mode
        }
        result(i, dense_col) = temp[0];
      }   
    }  

    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }  

  HB_EMUL_REG_KERNEL(tensorlib_sparse_dense_mm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
}
