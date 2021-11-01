//============================================================================
// for spmv kernels that require boundaries to be passed in block array
//============================================================================

#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>

extern "C" {
  __attribute__ ((noinline)) int tensorlib_block_generate (
    hb_tensor_t* _block,
    hb_tensor_t* _c2sr, //C2SR mode
    hb_tensor_t* _indices,
    hb_tensor_t* _values,
    int* _block_size) {
    
    auto block = HBTensor<int>(_block);
    auto c2sr = HBTensor<int>(_c2sr);  //C2SR mode
    auto indices = HBTensor<int>(_indices);
    auto values = HBTensor<float>(_values);
    int block_size = *_block_size;

    int m = block.numel() / block_size;
    //int32_t offset = m + 1;

    //int   *tmp_indices = (int*)indices.data_ptr();
    //float *tmp_values  = (float*)values.data_ptr();

    float thread_num = bsg_tiles_X * bsg_tiles_Y;
    float start = __bsg_id;
    float end = m;
    //int tag = 1;
 
    bsg_cuda_print_stat_kernel_start();
    //bsg_cuda_print_stat_start(tag);

    int val[block_size];

    for (int i = start; i < end; i = i + thread_num) {
        for (int j = 0; j < block_size; j++) {
            val[j] = c2sr(i);
        }
        for (int idx = c2sr(i); idx < c2sr(i + 1); idx++) {
            for (int j = 0; j < block_size; j++) {
                if (indices(idx) < block_size*(j+1)) {
                    val[j] += 1;
                }
            } 
        }
        for (int j = 0; j < block_size; j++) {
            block(m*j + i) = val[j];
        }
    }

    //bsg_cuda_print_stat_end(tag);
    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }  

  HB_EMUL_REG_KERNEL(tensorlib_block_generate, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, int*)
}