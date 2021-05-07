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

    auto r_indices = HBTensor<int>(_r_indices);
    auto r_values = HBTensor<float>(_r_values);
    auto t_indices = HBTensor<int>(_t_indices);
    auto t_values = HBTensor<float>(_t_values);
    auto s_indices = HBTensor<int>(_s_indices);
    auto s_values = HBTensor<float>(_s_values);
    auto result_nnz = HBTensor<int>(_result_nnz);
    
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    size_t end = t_values.numel();

    bsg_cuda_print_stat_kernel_start();
/*
    int nnz = 0;
    for (int t_val = start; t_val < end; t_val = t_val + thread_num) { // check nonzero val in t
      int row = t_indices(0, t_val);                           // get val's indices
      int col = t_indices(1, t_val);
      for (int s_val = 0; s_val < s_values.numel(); s_val++) { // search through values in s
        if (s_indices(0, s_val) == row && s_indices(1, s_val) == col) { // if indices match
          nnz++;                                                    // multiplication yields nnz
          break;
        }
      }
    }
    g_barrier.sync();

    if (nnz > 0) {
      int output_index = 0;                                    // initialize index variable
      for (int tile = 0; tile < __bsg_id; tile++) { 
        int x_coord = tile%bsg_tiles_X;
        int y_coord = static_cast<int>(tile / bsg_tiles_X);
        int *tile_nnz = reinterpret_cast<int*>(bsg_tile_group_remote_pointer(x_coord, y_coord, &nnz));
        output_index = output_index + *(tile_nnz);
      }
      if (__bsg_id == ((bsg_tiles_X*bsg_tiles_Y)-1)) {
        result_nnz(0) = output_index + nnz;
      }
      for (int t_val = start; t_val < end; t_val = t_val + thread_num) { // check nonzero val in t
        int row = t_indices(0, t_val);                           // get val's indices
        int col = t_indices(1, t_val);
        for (int s_val = 0; s_val < s_values.numel(); s_val++) { // search through values in s
          if (s_indices(0, s_val) == row && s_indices(1, s_val) == col) { // if indices match
            float elem_t = t_values(t_val);
            float elem_s = s_values(s_val);
            r_values(output_index) = elem_t * elem_s;
            r_indices(0, output_index) = row;
            r_indices(1, output_index) = col;
            output_index++;
            break;
          }
        }
      }
    }
*/ 
    bsg_cuda_print_stat_kernel_end(); 
    g_barrier.sync();  
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sparse_mul, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
}


