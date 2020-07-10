//====================================================================
// Dot product kernel
// 03/06/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_index_select(
          hb_tensor_t* self_p,
          hb_tensor_t* result_p,
          hb_tensor_t* index_p) {

    HBTensor<float> self(self_p);
    HBTensor<float> result(result_p);
    HBTensor<int32_t> index(index_p);
    float* self_data = (float*)self.data_ptr();
    float* result_data = (float*)result.data_ptr();
    int32_t* index_data = (int32_t*)index.data_ptr();

    bsg_cuda_print_stat_kernel_start();

    auto size0 = self.dim(0); // this should really be size(0)
    auto rowsize = size0 == 0 ? 1 : self.numel() / size0;

    if (rowsize > 0) {
      // self.ndim() can't be 0
      if (self.ndim() == 1) {
        hb_tiled_for(index.numel(), [&](size_t i) {
          hb_assert((index_data[i] >= 0 && index_data[i] < size0));
          result(i) = self(index_data[i]);
        });
      } else {
        hb_tiled_for(index.numel(), [&](size_t i) {
          memcpy(
            result_data + i * rowsize,
            self_data + index_data[i] * rowsize,
            rowsize * sizeof(float)
          );
        });
      }
    }

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_index_select, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
