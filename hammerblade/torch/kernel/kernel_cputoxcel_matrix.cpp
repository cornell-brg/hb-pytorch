//==========================================================================================
//The kernel code of changing the data layout of dense matrix from CPU to HB with SpMM Xcel
//20/02/2021 Zhongyuan Zhao(zz546@cornell.edu)
//==========================================================================================
#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_cputoxcel_matrix(
    hb_tensor_t* _dense_matrix,
    hb_tensor_t* _matrix) {

    auto dense_matrix = HBTensor<int>(_dense_matrix);
    auto matrix = HBTensor<int>(_matrix);
    uint32_t row = dense_matrix.dim(0);
    uint32_t col = dense_matrix.dim(1);
    int* m_ptr = (int*)matrix.data_ptr();
    int* dm_ptr = (int*)dense_matrix.data_ptr();

    uint32_t cacheline_word = CACHELINE_BYTE / 4; 
    uint32_t max_region_b = (((col + NUM_PE - 1) / NUM_PE) + cacheline_word - 1) / cacheline_word;
    
    uint32_t addr_b[NUM_PE]; 
    uint32_t b_index = 0;

    bsg_cuda_print_stat_kernel_start();
    if(__bsg_id < 16) {
      for(uint32_t i=0; i<max_region_b; i++) {
        for(uint32_t j=0; j < cacheline_word; j++){
          if(addr_b[__bsg_id] <  row * col) {
            m_ptr[b_index] = dm_ptr[addr_b[__bsg_id]];
          }
          else {
            m_ptr[b_index] = 0;
          }
          addr_b[__bsg_id]++;
          if(addr_b[__bsg_id] % col == 0) {
            addr_b[__bsg_id] += (NUM_PE - 1) * col;
          }
          b_index++;
        }
      } 
    }

    g_barrier.sync();
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }
  HB_EMUL_REG_KERNEL(tensorlib_cputoxcel_matrix, hb_tensor_t*, hb_tensor_t*)
}
