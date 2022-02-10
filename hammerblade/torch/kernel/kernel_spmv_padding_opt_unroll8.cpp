//============================================================================
// Sparse matrix multiply dense vector kernel
// Optimizations: non-blocking load, padding, C2SR format
// 04/05/2020 Zhongyuan Zhao (zz546@cornell.edu)
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

    int   *tmp_indices = (int*)indices.data_ptr();
    float *tmp_values  = (float*)values.data_ptr();
    float *tmp_vector  = (float*)vector.data_ptr();

    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
//    bsg_printf("Thread number is %d\n", thread_num);
    size_t start = __bsg_id;
    size_t end = m;
    uint32_t tag = 1;

    int indices_vcache = ((uintptr_t)tmp_indices / 128) % 32;
    int values_vcache = ((uintptr_t)tmp_values / 128) % 32;
    int indices_padding = (NUM_OF_SLOTS - indices_vcache) * CACHE_LINE;
    int values_padding = (NUM_OF_SLOTS - values_vcache) * CACHE_LINE;
//    if(__bsg_id == 0) {
//      bsg_printf("The vcache number of indices and values are %d, %d\n", indices_vcache, values_vcache);
//    }   
 
    bsg_cuda_print_stat_kernel_start();
    bsg_cuda_print_stat_start(tag);
//    bsg_printf("Tile %d is working\n", __bsg_id);
//    if(__bsg_id == 0) {
    for (uint32_t i = start; i < end; i = i + thread_num) {
      float temp = 0.0;
      int first = c2sr(offset + i);
      int last = first + c2sr(i + 1) - c2sr(i);
      int idx = first;
//      bsg_printf("First and last of row %d are %d and %d\n", i, first, last);
      for(; idx <= last - 8; idx = idx + 8) { //C2SR MODE
//        bsg_printf("Enter the unroll branch calculating the %d to %d nnz\n", idx, idx+7);
        int cidx0 = convert_idx(idx, m, i);
        int cidx1 = convert_idx(idx+1, m, i);
        int cidx2 = convert_idx(idx+2, m, i);
        int cidx3 = convert_idx(idx+3, m, i);
        int cidx4 = convert_idx(idx+4, m, i);
        int cidx5 = convert_idx(idx+5, m, i);
        int cidx6 = convert_idx(idx+6, m, i);
        int cidx7 = convert_idx(idx+7, m, i);
    
        asm volatile("": : :"memory");
        register int index0 = *(tmp_indices + cidx0 + indices_padding);
        register int index1 = *(tmp_indices + cidx1 + indices_padding);
        register int index2 = *(tmp_indices + cidx2 + indices_padding);
        register int index3 = *(tmp_indices + cidx3 + indices_padding);
        register int index4 = *(tmp_indices + cidx4 + indices_padding);
        register int index5 = *(tmp_indices + cidx5 + indices_padding);
        register int index6 = *(tmp_indices + cidx6 + indices_padding);
        register int index7 = *(tmp_indices + cidx7 + indices_padding);

        register float tmp_smatrix0 = *(tmp_values + cidx0 + values_padding);
        register float tmp_smatrix1 = *(tmp_values + cidx1 + values_padding);
        register float tmp_smatrix2 = *(tmp_values + cidx2 + values_padding);
        register float tmp_smatrix3 = *(tmp_values + cidx3 + values_padding);
        register float tmp_smatrix4 = *(tmp_values + cidx4 + values_padding);
        register float tmp_smatrix5 = *(tmp_values + cidx5 + values_padding);
        register float tmp_smatrix6 = *(tmp_values + cidx6 + values_padding);
        register float tmp_smatrix7 = *(tmp_values + cidx7 + values_padding);
        
  //      asm volatile("": : :"memory");
        register float tmp_dvector0 = *(tmp_vector + index0);
        register float tmp_dvector1 = *(tmp_vector + index1);
        register float tmp_dvector2 = *(tmp_vector + index2);
        register float tmp_dvector3 = *(tmp_vector + index3);
        register float tmp_dvector4 = *(tmp_vector + index4);
        register float tmp_dvector5 = *(tmp_vector + index5);
        register float tmp_dvector6 = *(tmp_vector + index6);
        register float tmp_dvector7 = *(tmp_vector + index7);
        asm volatile("": : :"memory");      
        temp = temp + tmp_smatrix0 * tmp_dvector0 + tmp_smatrix1 * tmp_dvector1
                    + tmp_smatrix2 * tmp_dvector2 + tmp_smatrix3 * tmp_dvector3
                    + tmp_smatrix4 * tmp_dvector4 + tmp_smatrix5 * tmp_dvector5
                    + tmp_smatrix6 * tmp_dvector6 + tmp_smatrix7 * tmp_dvector7;       
       
      }
      for(; idx < last; idx++) {
//        bsg_printf("Enter the normal branch calculating the %d nnz\n", idx);
        int cidx = convert_idx(idx, m, i);
        register int index = *(tmp_indices + cidx + indices_padding);
        register float tmp_smatrix = *(tmp_values + cidx + values_padding);
        register float tmp_dvector = *(tmp_vector + index);
        asm volatile("": : :"memory");
        temp = temp + tmp_smatrix * tmp_dvector;
      }
      result(i) = temp;
//      bsg_printf("Row %d is caculated\n", i);
    }
//    }
//    bsg_printf("Row %d finish computing !\n", i);
    bsg_cuda_print_stat_end(tag);
    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }  

  HB_EMUL_REG_KERNEL(tensorlib_spmv, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
}
