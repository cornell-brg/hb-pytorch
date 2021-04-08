//====================================================================
// _cat kernel
//====================================================================
// _cat kernel support cat on any dim
//
// Zhonguan Zhao
// Date : 03/01/2021

#include <kernel_common.hpp>

extern "C" {

//====================================================================
// tensorlib__cat
//====================================================================
// This is a simple _cat kernel only works with 0 dim

__attribute__ ((noinline))
int tensorlib__cat( hb_tensor_t** tensors_p, hb_tensor_t* result_p,
                    uint32_t* length_p, int32_t* dim_p)
{
  HBTensor<float> result(result_p);
  float* result_data = (float*)result.data_ptr();
  uint32_t length = *length_p;
  int32_t dim = *dim_p;
  
  int outer = 1;
  int inner = 1;

  for(int i = 0; i < dim; i++) {
    outer *= result.dim(i);
  }

  for(int i = dim + 1; i < result.ndim(); i++) {
    inner *= result.dim(i);
  }

  size_t num_threads = bsg_tiles_X * bsg_tiles_Y;
  int offset = 0;
  
  bsg_cuda_print_stat_kernel_start();
  bsg_saif_start();

  int offset_strides = 0;
  for(int j = 0; j < length; j++) {
    HBTensor<float> input(tensors_p[j]);
    int local_inner = inner * input.dim(dim);
    offset_strides += local_inner;
  }

  for(int o = __bsg_id; o < outer; o = o + num_threads) {
    offset = o * offset_strides;
    for(int j = 0; j < length; j++) {
      HBTensor<float> input(tensors_p[j]);
      float *input_data = (float*)input.data_ptr();
      int local_inner = inner * input.dim(dim);
      float *result_ptr = result_data + offset;
      float *input_ptr = input_data + o * local_inner;
      int idx = 0;
      for(; idx <= local_inner - 8; idx++) {
        *(result_ptr + idx) = *(input_ptr + idx);
        *(result_ptr + idx + 1) = *(input_ptr + idx + 1);
        *(result_ptr + idx + 2) = *(input_ptr + idx + 2);
        *(result_ptr + idx + 3) = *(input_ptr + idx + 3);
        *(result_ptr + idx + 4) = *(input_ptr + idx + 4);
        *(result_ptr + idx + 5) = *(input_ptr + idx + 5);
        *(result_ptr + idx + 6) = *(input_ptr + idx + 6);
        *(result_ptr + idx + 7) = *(input_ptr + idx + 7);
      }
      for(; idx < local_inner; idx++) {
        *(result_ptr + idx) = *(input_ptr + idx);
      }
      offset += local_inner;
    }
  }
  
  bsg_saif_end();
  bsg_cuda_print_stat_kernel_end();
  g_barrier.sync();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib__cat, hb_tensor_t**, hb_tensor_t*, uint32_t*, int32_t*)

}
