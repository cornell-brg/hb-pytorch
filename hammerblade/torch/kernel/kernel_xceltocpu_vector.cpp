//==========================================================================================
//The kernel code of changing the data layout of dense vector from HB with SpMV Xcel back to CPU
//11/07/2020 Zhongyuan Zhao(zz546@cornell.edu)
//==========================================================================================
#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_xceltocpu_vector(
    hb_tensor_t* _result,
    hb_tensor_t* _xcel_out) {
    auto result = HBTensor<int>(_result);
    auto xcel_out = HBTensor<int>(_xcel_out);
    int* r = (int*)result.data_ptr();
    int* xcel = (int*)xcel_out.data_ptr();
    
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    uint32_t n = result.numel();
    uint32_t cacheline_word = CACHELINE_BYTE / 4;
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();
    for(uint32_t i = __bsg_id; i < n; i = i + thread_num) {
      uint32_t index = i % NUM_PE;
      uint32_t offset = i / NUM_PE;
      uint32_t addr = index * cacheline_word + (offset / cacheline_word) * cacheline_word * NUM_PE + (offset % cacheline_word);
      r[i] = xcel[addr];
    }

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }
  HB_EMUL_REG_KERNEL(tensorlib_xceltocpu_vector, hb_tensor_t*, hb_tensor_t*)
}
