//==============================================================================================
// Sparse matrix multiply dense vector kernel
// This kernel has two shared memory and each group calculates partial sum with half of vector
//==============================================================================================

#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>
extern "C" {
  __attribute__ ((noinline)) int tensorlib_spmv_SP1(
    hb_tensor_t* _result,
    hb_tensor_t* _c2sr, //C2SR mode
    hb_tensor_t* _indices,
    hb_tensor_t* _values,
    hb_tensor_t* _dense_vector,
    hb_tensor_t* _block,
    int* _block_size) {
    
    auto result = HBTensor<float>(_result);
    auto c2sr = HBTensor<int>(_c2sr);  //C2SR mode
    auto indices = HBTensor<int>(_indices);
    auto values = HBTensor<float>(_values);
    auto vector = HBTensor<float>(_dense_vector);
    auto block = HBTensor<int>(_block);
    int block_size = *_block_size;
    int m = result.numel();
    int n = vector.numel();
    //int32_t offset = m + 1;
    int m_bl = m/4;
    int n_bl = n/32;

    int thread_num = bsg_tiles_X * bsg_tiles_Y;

    m_bl = 250;
    int m_num = m/250;
    if (m%250 != 0) {
      m_num += 1;
    }
    
    float local_B[n_bl];
    size_t sp_start = (n_bl)*(__bsg_id%32);
    size_t sp_end = sp_start + n_bl;
    sp_end = sp_end < n ? sp_end : n;

    float partial_out[m_bl]; //change number accordingly

    int tag = 1;
    int i, j, col_idx, B_idx;

    bsg_cuda_print_stat_kernel_start();
    //bsg_cuda_print_stat_start(tag);

    float *p_o[32];

    if (__bsg_id % 32 == 0) {
      for (i = 1; i < 32; i ++) {
        if (i < 16) {
          p_o[i] = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x + i, bsg_y, &partial_out[0]));
        }
        else if (i >= 16) {
          p_o[i] = reinterpret_cast<float*>(bsg_tile_group_remote_pointer((bsg_x + (i-16)), bsg_y + 1, &partial_out[0]));
        }
      }
    }
    g_barrier.sync();
    //bsg_tile_group_barrier(&r_barrier, &c_barrier); 

    //memcpy dense vector
    for (i = sp_start; i < sp_end; i ++) {
      local_B[j] = vector(i);
      j++;
    }

    float temp;
    int start, end;

    for (int k = 0; k < m_num; k++) {
      start = __bsg_id/32 * m_bl * (k + 1);
      end = start + m_bl * (k + 1);
      end = end < m ? end : m;

      //multiply
      if (__bsg_id%32 == 0) {
        for (i = start; i < end; i++) {
          temp = 0;
          for (col_idx = c2sr(i); col_idx < block(i); col_idx ++) {
            B_idx = indices(col_idx) - sp_start;
            temp += values(col_idx) * local_B[B_idx];
          }
          partial_out[i - start] = temp;
        }
      }
      else if (__bsg_id%32 == 31) {
        for (i = start; i < end; i++) {
          temp = 0;
          for (col_idx = block(m*30 + i); col_idx < c2sr(i+1); col_idx ++) {
            B_idx = indices(col_idx) - sp_start;
            temp += values(col_idx) * local_B[B_idx];
          }
          partial_out[i - start] = temp;
        }
      }
      else if ((__bsg_id%32 > 0) && (__bsg_id%32 < 31)) {
        for (i = start; i < end; i++) {
          temp = 0;
          for (col_idx = block(m*(__bsg_id%32 - 1) + i); col_idx < block(m*(__bsg_id%32) + i); col_idx ++) {
            B_idx = indices(col_idx - sp_start);
            temp += values(col_idx) * local_B[B_idx];
          }
          partial_out[i - start] = temp;
        }
      }
      //bsg_tile_group_barrier(&r_barrier, &c_barrier); 
      g_barrier.sync();

      //add to output vector
      if (__bsg_id % 32 == 0) {
        for (i = 0; i < m_bl; i ++) {
          for (j = 1; j < 32; j ++) {
            partial_out[i] += *(p_o[j] + i);
          }
          result(i + start) = partial_out[i];
        }
      }
    }
    
    //bsg_tile_group_barrier(&r_barrier, &c_barrier); 
    g_barrier.sync();

    //bsg_cuda_print_stat_end(tag);
    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }  

  HB_EMUL_REG_KERNEL(tensorlib_spmv_SP1, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, int*)
}
