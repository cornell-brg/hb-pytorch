//====================================================================
// Upsample Nearest 1d kernel
// 4/26/2022 Aditi Agarwal (aa2224@cornell.edu)
//====================================================================


#include <kernel_common.hpp>
#include <cmath>
#include <math.h>

extern "C" {

//taken from aten/src/ATen/native/UpSample.h
static inline int64_t nearest_neighbor_compute_source_index(
    const float scale,
    int64_t dst_index,
    int64_t input_size) {
  const int64_t src_index =
      std::min(static_cast<int64_t>(floorf(dst_index * scale)), input_size - 1);
  return src_index;
}

  
  __attribute__ ((noinline)) static int tensorlib_upsample_nearest1d(
          int64_t* odata,
          int64_t* idata,
          int64_t input_width,
          int64_t output_width,
          int64_t nbatch,
          int64_t channels) {
    
    // auto input = HBTensor<float>(t0_p);
    const float scale = (float)input_width / (float)output_width;
    channels = channels * nbatch;



    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();
    
      // special case: just copy
    if (input_width == output_width) {
      hb_tiled_for(output_width, [&](int64_t w2) {
        const int64_t w1 = w2;
        const int64_t* pos1 = &idata[w1];
        int64_t* pos2 = &odata[w2];

        for (int64_t c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += input_width;
          pos2 += output_width;
        }
      });
      return 0;
    }
    hb_tiled_for(output_width, [&](int64_t w2) {
      const int64_t src_x = nearest_neighbor_compute_source_index(scale, w2, input_width);
      const int64_t w1 = src_x;
      const int64_t* pos1 = &idata[w1];
      int64_t* pos2 = &odata[w2];

      for (int64_t c = 0; c < channels; ++c) {
        pos2[0] = pos1[0];
        pos1 += input_width;
        pos2 += output_width;
      }
    });

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_upsample_nearest1d, int64_t*, int64_t*, int64_t, int64_t, int64_t, int64_t)

}
