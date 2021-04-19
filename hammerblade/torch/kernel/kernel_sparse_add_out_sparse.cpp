//=======================================================================
// Sparse tensor add sparse tensor kernel
// 04/16/2020 Zhongyuan Zhao (zz546@cornell.edu) Anya Probowo
//=======================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_sparse_add_out_sparse(
          hb_tensor_t* _r_indices,
          hb_tensor_t* _r_values,
          hb_tensor_t* _t_indices,
          hb_tensor_t* _t_values,
          hb_tensor_t* _s_indices,
          hb_tensor_t* _s_values,
          hb_tensor_t* _result_nnz,
          hb_tensor_t* _n,
          float* _alpha) {

    auto r_indices = HBTensor<float>(_r_indices);
    auto r_values = HBTensor<float>(_r_values);
    auto t_indices = HBTensor<int>(_t_indices);
    auto t_values = HBTensor<float>(_t_values);
    auto s_indices = HBTensor<int>(_s_indices);
    auto s_values = HBTensor<float>(_s_values);
    auto result_nnz = HBTensor<float>(_result_nnz);
    auto n = HBTensor<int>(_n);
    float alpha= *_alpha; 
    
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    result_nnz = 0;

    bsg_cuda_print_stat_kernel_start();

    /*TODO: Add your code here:

    int tid_x = __bsg_tile_group_id_x * BSG_TILE_GROUP_X_DIM + __bsg_x;
    int tid_y = __bsg_tile_group_id_y * BSG_TILE_GROUP_Y_DIM + __bsg_y;
    int num_threads_x = BSG_TILE_GROUP_X_DIM * __bsg_grid_dim_x;
    int num_threads_y = BSG_TILE_GROUP_Y_DIM * __bsg_grid_dim_y;

    int num_threads = num_threads_x * num_threads_y;
    int tid = tid_y * num_threads_x + tid_x;
    */

    int nnz = 0; // number of nonzero values calculated by this tile

    // binary search for non zeros of matrix s overlapping with matrix t
    for (int idx_A = start; idx_A < s_values.numel(); idx_A += thread_num) {

      int row = s_indices(0, idx_A); // row for value idx_A
      int col = s_indices(1, idx_A); // col for value idx_A
      int idx_A_abs = row*n + col;   // absolute index for value idx_A
      int l = 0;                     // left index
      int r = t_values.numel();      // right index (nnz for matrix t)
      int check = 0;

      // while left index is less than right index and 
      while ((l <= r) && (check == 0)) { 
        int m = l + (r - l) / 2; // middle index
        int idx_B_abs = t_indices(0, m)*n + t_indices(1, m);

        // check if x is present at mid 
        if (idx_B_abs == idx_A_abs) {
          data_t sum_AB = s_values(idx_A) + t_values(m);
          if (sum_AB != 0) {
            nnz++;
            check = 1;
          }
        // if idx_A_abs greater, ignore left half 
        } else if (idx_B_abs < idx_A_abs) {
          l = m + 1; 
        // if idx_A_abs is smaller, ignore right half 
        } else {
          r = m - 1; 
        }
      }
      if (!check) {
        nnz++;
      }
    }

    // binary search for non zeros of matrix t overlapping with matrix s
    for (int idx_B = start; idx_B < t_values.numel(); idx_B += thread_num) { // tile checks a nonzero value in A

      int row = t_indices(0, idx_B);                  // get value's indices
      int col = t_indices(1, idx_B);
      int idx_B_abs = row*n + col;                             // absolute index of A value
      int l = 0;
      int r = s_values.numel();
      int check = 0;

      while ((l <= r) && (check == 0)) { 
        // middle index
        int m = l + (r - l) / 2;
        int idx_A_abs = s_indices(0, m)*n + s_indices(1, m);

        // check if x is present at mid 
        if (idx_A_abs == idx_B_abs) {
          data_t sum_AB = s_values(m) + t_values(idx_B);
          if (sum_AB != 0) {
            check = 1;
          }
        // if idx_B_abs greater, ignore left half 
        } else if (idx_A_abs < idx_B_abs) {
          l = m + 1; 
        // if idx_B_abs is smaller, ignore right half 
        } else {
          r = m - 1; 
        }
      }
      if (!check) {
        nnz++;
      }
    }

    // all tiles should finish their computations before proceeding
    g_barrier.sync();
    
    // add all the nnz to return in result_nnz
    for (int i = 0; i < thread_num; i++) {
      int x_coord = tile%BSG_TILE_GROUP_X_DIM;      // get tile coordinates
      int y_coord = static_cast<int>(i / BSG_TILE_GROUP_X_DIM);
      int *tile_nnz = reinterpret_cast<int*>(bsg_tile_group_remote_pointer(x_coord, y_coord, &nnz));
      result_nnz += *(tile_nnz);
    }

    if (nnz > 0) {                                    // if tile computed any nonzero values
      int output_index = 0;                           // starting index in Out that this tile should write nnz values to
      for (int tile = 0; tile < start; tile++) {        // for each tile with a start less than this tile
        int x_coord = tile%BSG_TILE_GROUP_X_DIM;      // get tile coordinates
        int y_coord = static_cast<int>(tile / BSG_TILE_GROUP_X_DIM);
        int *tile_nnz = reinterpret_cast<int*>(bsg_tile_group_remote_pointer(x_coord, y_coord, &nnz));
        output_index = output_index + *(tile_nnz);
      }

      for (int idx_A = start; idx_A < s_values.numel(); idx_A += thread_num) { // tile checks a nonzero value in A
        int row = s_indices(0, idx_A);                  // get value's indices
        int col = s_indices(1, idx_A);
        int idx_A_abs = row*n + col;                             // absolute index of A value
        int l = 0;
        int r = t_values.numel();
        int check = 0;

        while ((l <= r) && (check == 0)) { 
          // middle index
          int m = l + (r - l) / 2; 
          int idx_B_abs = t_indices(0, m)*n + t_indices(1, m);
          
          // check if x is present at mid 
          if (idx_B_abs == idx_A_abs) {
            data_t sum_AB = s_values(idx_A) + t_values(m);
            if (sum_AB != 0) {
              r_indices(0,output_index) = row;
              r_indices(1,output_index) = col;
              r_values(output_index) = sum_AB;
              output_index += 1;
              check = 1;
            }
          // if idx_A_abs greater, ignore left half 
          } else if (idx_B_abs < idx_A_abs) {
            l = m + 1; 
          // if idx_A_abs is smaller, ignore right half 
          } else {
            r = m - 1; 
          }
        }
        if (!check) {
          r_indices(0,output_index) = row;
          r_indices(1,output_index) = col;
          r_values(output_index) = s_values(idx_A);
          output_index += 1;
        }
      }

      for (int idx_B = start; idx_B < t_values.numel(); idx_B += thread_num) { // tile checks a nonzero value in A
        int row = t_indices(0, idx_B);                  // get value's indices
        int col = t_indices(1, idx_B);
        int idx_B_abs = row*n + col;                             // absolute index of A value
        int l = 0;
        int r = s_values.numel();
        int check = 0;

        while ((l <= r) && (check == 0)) { 
          int m = l + (r - l) / 2;
          int idx_A_abs = s_indices(0, m)*n + s_indices(1, m);

          // check if x is present at mid 
          if (idx_A_abs == idx_B_abs) {
            data_t sum_AB = s_values(m) + t_values(idx_B);
            if (sum_AB != 0) {
              check = 1;
            }
          // if abs_idx greater, ignore left half 
          } else if (idx_A_abs < idx_B_abs) {
            l = m + 1; 
          // if abs_idx is smaller, ignore right half 
          } else {
            r = m - 1; 
          }
        }
        if (!check) {
          r_indices(0,output_index) = row;
          r_indices(1,output_index) = col;
          r_values(output_index) = t_values(idx_B);
          output_index += 1;
        }
      }
    }

    bsg_cuda_print_stat_kernel_end(); 
    g_barrier.sync();  
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sparse_add_out_sparse, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*)
}