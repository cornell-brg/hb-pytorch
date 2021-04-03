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

    char tid_map[8][16] = {
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 128, 128},
      {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 128, 128},
      {28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 128, 128},
      {42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 128, 128},
      {56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 128, 128},
      {70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 128, 128},
      {84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 128, 128},
      {98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 128, 128},
    };
    
    char (&mc_config)[8][16] = tid_map;
    int tile_id = (int)mc_config[bsg_y][bsg_x];
    int thread_num = bsg_tiles_X * bsg_tiles_Y; 

    uint32_t cacheline_word = CACHELINE_BYTE / 4;
    uint32_t out_ptr[NUM_PE];
   
    bsg_cuda_print_stat_kernel_start();
    if(tile_id < 128) {
      for(int i = tile_id; i < m; i = i + thread_num) {
        for(int j = 0; j < k; j++) {
          int tmp = i / NUM_PE;
          int offset = tmp * k + j;
          int seg_offset = offset % cacheline_word;
          int seg1 = offset / cacheline_word;
          int seg2 = i % NUM_PE;
          int outm_offset = seg1 * NUM_PE * cacheline_word + seg2 * cacheline_word + seg_offset;
          result(i, j) = xcel[outm_offset];
        }
      }
    }
    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();

    return 0;
  }
  HB_EMUL_REG_KERNEL(tensorlib_xceltocpu_matrix, hb_tensor_t*, hb_tensor_t*)
}
