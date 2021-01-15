//========================================================================
// Upsampling 1D with nearest algo
//========================================================================
// Authors  : Lin Cheng
// Date     : 01/06/2021

#include <kernel_common.hpp>

static inline int32_t nearest_neighbor_compute_source_index(
    const float scale,
    int32_t dst_index,
    int32_t input_size) {
  const int32_t src_index =
      std::min(static_cast<int32_t>(floorf(dst_index * scale)), input_size - 1);
  return src_index;
}

extern "C" {

__attribute__ ((noinline))
int tensorlib_upsample_nearest1d(
    hb_tensor_t* output_p,
    hb_tensor_t* input_p,
    int32_t*     input_width_p,
    int32_t*     output_width_p,
    int32_t*     nbatch_p,
    int32_t*     channels_p ) {

  HBTensor<float> output(output_p);
  HBTensor<float> input(input_p);
  int32_t         input_width = *input_width_p;
  int32_t         output_width = *output_width_p;
  int32_t         nbatch = *nbatch_p;
  int32_t         channels = *channels_p;

  float* idata = (float*)input.data_ptr();
  float* odata = (float*)output.data_ptr();

  const float scale = (float)input_width / (float)output_width;
  channels = channels * nbatch;

  bsg_cuda_print_stat_start(2);

  for (int64_t c = 0; c < channels; ++c) {
    hb_tiled_for(output_width, [&](size_t w2) {
      const int32_t w1 =
          nearest_neighbor_compute_source_index(scale, w2, input_width);
      const float* pos1 = &idata[w1];
      float* pos2 = &odata[w2];
      pos2[0] = pos1[0];
    });
    idata += input_width;
    odata += output_width;
  }

  bsg_cuda_print_stat_end(2);
  g_barrier.sync();
  return 0;

}

HB_EMUL_REG_KERNEL(tensorlib_upsample_nearest1d, hb_tensor_t*, hb_tensor_t*,
                   int32_t*, int32_t*, int32_t*, int32_t*)

__attribute__ ((noinline))
int tensorlib_upsample_nearest1d_back(
    hb_tensor_t* output_p,
    hb_tensor_t* input_p,
    int32_t*     input_width_p,
    int32_t*     output_width_p,
    int32_t*     nbatch_p,
    int32_t*     channels_p ) {

  HBTensor<float> output(output_p);
  HBTensor<float> input(input_p);
  int32_t         input_width = *input_width_p;
  int32_t         output_width = *output_width_p;
  int32_t         nbatch = *nbatch_p;
  int32_t         channels = *channels_p;

  float* idata = (float*)input.data_ptr();
  float* odata = (float*)output.data_ptr();

  const float scale = (float)input_width / (float)output_width;
  channels = channels * nbatch;

  //for (int32_t w1 = 0; w1 < input_width; ++w1) {
  bsg_cuda_print_stat_start(3);

  for (int64_t c = 0; c < channels; ++c) {
    hb_tiled_for(input_width, [&](size_t w1) {
      int32_t start_idx = w1 / scale;
      int32_t end_idx = start_idx + (1 / scale);
      end_idx = end_idx > output_width ? output_width : end_idx;

      float* pos1 = &idata[w1];
      float sum = 0;
      for (int32_t w2 = start_idx; w2 < end_idx; ++w2) {
        const float* pos2 = &odata[w2];
        sum += pos2[0];
      }
      pos1[0] += sum;
    });
    idata += input_width;
    odata += output_width;
  }

  bsg_cuda_print_stat_end(3);
  g_barrier.sync();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib_upsample_nearest1d_back, hb_tensor_t*, hb_tensor_t*,
                   int32_t*, int32_t*, int32_t*, int32_t*)

}
