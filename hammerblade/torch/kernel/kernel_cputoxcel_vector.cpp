//==========================================================================================
//The kernel code of changing the data layout of dense vector from CPU to HB with SpMV Xcel
//11/07/2020 Zhongyuan Zhao(zz546@cornell.edu)
//==========================================================================================
#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_cputoxcel_vector(
    hb_tensor_t* _dense_vector,
    hb_tensor_t* _vector) {

    auto dense_vector = HBTensor<int>(_dense_vector);
    auto xcel_vector = HBTensor<int>(_vector);
    uint32_t col = dense_vector.numel();
    int* xv_ptr = (int*)xcel_vector.data_ptr();
    int* dv_ptr = (int*)dense_vector.data_ptr();

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
    uint32_t thread_num = bsg_tiles_X  * bsg_tiles_Y;
    uint32_t cacheline_word = CACHELINE_BYTE / 4; 
    uint32_t region = NUM_PE * cacheline_word;
      
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    if(tile_id < 128) {
//      bsg_printf("tile_id=%d, [bsg_x=%d, bsg_y=%d], [bsg_tiles_X=%d, bsg_tiles_Y=%d]\n", tile_id, bsg_x, bsg_y, bsg_tiles_X, bsg_tiles_Y);
      for(int i = tile_id; i < col; i = i + thread_num) {
        int seg1 = i / region;
        int tmp = i % region;
        int seg2 = tmp % NUM_PE;
        int local_offset = tmp /NUM_PE;
        xv_ptr[seg1 * region + seg2 * cacheline_word + local_offset] = dv_ptr[i];
      }
    }

    bsg_saif_end();   
//    g_barrier.sync();
    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }
  HB_EMUL_REG_KERNEL(tensorlib_cputoxcel_vector, hb_tensor_t*, hb_tensor_t*)
}
