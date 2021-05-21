//====================================================================
// _cat kernel
//====================================================================
// simple _cat kernel only works with 0 dim
//
// Authors : Lin Cheng, Jack Weber
// Date    : 07/29/2020, 08/04/2020

#define BUF_SIZE 16
#include <kernel_common.hpp>

extern "C" {

//====================================================================
// tensorlib__cat
//====================================================================
// This is a simple _cat kernel only works with 0 and 1 dim

__attribute__ ((noinline))
int tensorlib__cat( hb_tensor_t** tensors_p, hb_tensor_t* result_p,
                    uint32_t* length_p, int32_t* dim_p)
{
  HBTensor<float> result(result_p);
  uint32_t length = *length_p;
  hb_assert(length <= BUF_SIZE);
  uint32_t dim = *dim_p;
  int32_t arr[BUF_SIZE];
  int32_t dim0[BUF_SIZE];
  int32_t dim1[BUF_SIZE];

  // collect tensors' size
  for(size_t i = 0; i < length; i++) {
    HBTensor<float> tensor(tensors_p[i]);
    arr[i] = tensor.numel();
    dim0[i] = tensor.dim(0);
    int32_t n = tensor.ndim();
    if (n>1){
      dim1[i] = tensor.dim(1);
    }
  }
  
  bsg_cuda_print_stat_kernel_start(); 
  hb_tiled_for(result.numel(), [&] (int32_t i) {
    int32_t j = 0;
    int32_t index = 0;
    int32_t size = arr[0];  
    int32_t q = 0;    
   
    if (dim == 1){
      size = dim0[0]*dim1[0];
    }

    while (i >= size) {
      index = i - size;
      j++;

      if (dim == 1){
        q = j%length;
        size += dim0[q]*dim1[q];
      }
      else{
        q = j;
        size += arr[j];
      }
    }
    if (j == 0) {
      index = i;
    }
    if (j>=length && dim == 1){
      index = index + dim0[q]*dim1[q];
    }

    HBTensor<float> t(tensors_p[q]);
    result(i) = t(index);
  });

  bsg_saif_end();
  bsg_cuda_print_stat_kernel_end();
  g_barrier.sync();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib__cat, hb_tensor_t**, hb_tensor_t*, uint32_t*, int32_t*)

}
