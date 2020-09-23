//============================================================================
// Sparse matrix multiply dense matrix kernel
// 04/05/2020 Zhongyuan Zhao, Michael Rivera (zz546@cornell.edu)
//============================================================================

#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>
extern "C" {

  __attribute__ ((noinline)) int tensorlib_spmv(
    hb_tensor_t* _result,
    hb_tensor_t* _c2sr, //C2SR mode
    hb_tensor_t* _indices,
    hb_tensor_t* _values,
    hb_tensor_t* _dense_vector) {
    
    auto result = HBTensor<float>(_result);
    auto c2sr = HBTensor<int>(_c2sr);  //C2SR mode
    auto indices = HBTensor<int>(_indices);
    auto values = HBTensor<float>(_values);
    auto vector = HBTensor<float>(_dense_vector);
    uint32_t m = result.numel();
    int32_t offset = m + 1;

    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    size_t end = m;
    uint32_t tag = 1;
    
    bsg_cuda_print_stat_kernel_start();
    bsg_cuda_print_stat_start(tag);
//    printf("Enter the spmv kernel\n");

    for (uint32_t i = start; i < end; i = i + thread_num) {
      float temp = 0.0;
      uint32_t first = c2sr(offset + i);
      uint32_t last = c2sr(offset + i) + c2sr(i + 1) - c2sr(i);
      for(uint32_t idx = first; idx < last; idx++) { //C2SR MODE
       uint32_t col_index = convert_idx(idx, m, i);
//       printf("idx is %d, m is %d, i is %d and col_index is %d\n", idx, m, i, col_index);
//       printf("indices(%d) is %d\n", col_index, indices(col_index));
       temp = temp + values(col_index) * vector(indices(col_index)); //C2SR mode
      }
      result(i) = temp;
    }  

    bsg_cuda_print_stat_end(tag);
    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }  

  HB_EMUL_REG_KERNEL(tensorlib_spmv, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
}
