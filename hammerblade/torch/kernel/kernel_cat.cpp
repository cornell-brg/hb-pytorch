//====================================================================
// _cat kernel
//====================================================================
// simple _cat kernel only works with 0 dim
//
// Authors : Lin Cheng, Janice Wei
// Date    : 07/29/2020, 08/04/2020


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
  uint32_t length = *length_p;
  int32_t dim = *dim_p;

  bsg_cuda_print_stat_kernel_start();

  hb_tiled_for(result.numel(), [&] (int32_t i) {
	int32_t j = 0;
	int32_t index = 0;
	HBTensor<float> init(tensors_p[0]);
	int32_t size = init.numel();
    while (i >= size) {
      index = i - size;
      HBTensor<float> tensor(tensors_p[++j]);
      size += tensor.numel();
    }
    if (j == 0) {
      index = i;
    }
    HBTensor<float> t(tensors_p[j]);
    result(i) = t(index);
  });

  bsg_cuda_print_stat_kernel_end();
  g_barrier.sync();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib__cat, hb_tensor_t**, hb_tensor_t*, uint32_t*, int32_t*)

}
