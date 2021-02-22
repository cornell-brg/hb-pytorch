//==========================================================================================
//The kernel code of changing the data layout of dense matrix from HB with SpMM Xcel back to CPU
//20/02/2021 Zhongyuan Zhao(zz546@cornell.edu)
//==========================================================================================
#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_xceltocpu_matrix(
    hb_tensor_t* _result,
    hb_tensor_t* _xcel_out) {
    auto result = HBTensor<int>(_result);
    auto xcel_out = HBTensor<int>(_xcel_out);
    int m = result.dim(0);
    int k = result.dim(1);
    int* r = (int*)result.data_ptr();
    int* xcel = (int*)xcel_out.data_ptr();
    
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    uint32_t cacheline_word = CACHELINE_BYTE / 4;
    uint32_t out_ptr[NUM_PE];
    for(int i = 0; i < NUM_PE; i++){
      out_ptr[i] = i * cacheline_word;
    }   
    bsg_cuda_print_stat_kernel_start();
    for(int i = __bsg_id; i < m; i = i + thread_num) {
      for(int j = 0; j < k; j++) {
        r[i*k+j] = xcel[out_ptr[i % NUM_PE]];
        out_ptr[i % NUM_PE]++;
        if(out_ptr[i % NUM_PE] % cacheline_word == 0) {
          out_ptr[i % NUM_PE] += (NUM_PE - 1) * cacheline_word;
        } 
      }
    }
    g_barrier.sync();
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }
  HB_EMUL_REG_KERNEL(tensorlib_xceltocpu_matrix, hb_tensor_t*, hb_tensor_t*)
}
