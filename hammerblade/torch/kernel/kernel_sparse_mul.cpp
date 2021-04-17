//=======================================================================
// Sparse tensor mul sparse tensor kernel
// 04/16/2020 Zhongyuan Zhao (zz546@cornell.edu) Morgan Cupp
//=======================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_sparse_mul(
          hb_tensor_t* _r_indices,
          hb_tensor_t* _r_values,
          hb_tensor_t* _t_indices,
          hb_tensor_t* _t_values,
          hb_tensor_t* _s_indices,
          hb_tensor_t* _s_values,
          hb_tensor_t* _result_nnz) {

    auto r_indices = HBTensor<float>(_r_indices);
    auto r_values = HBTensor<float>(_r_values);
    auto t_indices = HBTensor<int>(_t_indices);
    auto t_values = HBTensor<float>(_t_values);
    auto s_indices = HBTensor<int>(_s_indices);
    auto s_values = HBTensor<float>(_s_values);
    auto result_nnz = HBTensor<float>(_result_nnz);
    
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;

    bsg_cuda_print_stat_kernel_start();

    /*TODO: Add your code here:
    */ 

    bsg_cuda_print_stat_kernel_end(); 
    g_barrier.sync();  
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sparse_mul, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
}


