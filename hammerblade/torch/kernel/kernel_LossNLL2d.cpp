//========================================================================
// NLL Loss 2d
//========================================================================
// Authors  : Lin Cheng
// Date     : 01/08/2021

#include <kernel_common.hpp>
#include <hb_reduction.hpp>

extern "C" {

// XXX: in this implementation weight is assumed to be nullptr
__attribute__ ((noinline))
int tensorlib_nll_loss2d_forward(
               hb_tensor_t* output_p,
               hb_tensor_t* total_weight_p,
               hb_tensor_t* input_p,
               hb_tensor_t* target_p,
               uint32_t*     reduction_p,
               int32_t*     ignore_index_p) {

  HBTensor<float> output(output_p);
  HBTensor<float> total_weight(total_weight_p);
  HBTensor<float> input(input_p);
  HBTensor<float> target(target_p);
  uint32_t reduction = *reduction_p;
  int32_t ignore_index = *ignore_index_p;

  const auto n_classes = input.dim(1);

  if (reduction == Reduction::None) {
    // XXX: unimplemented
    hb_assert(false);
    return 0;
  }

  const float* input_data    = (float*)input.data_ptr();
  const int64_t* target_data = (int64_t*)target.data_ptr();

  const uint32_t batch_size = input.dim(0);
  const uint32_t map_size = input.dim(2) * input.dim(3);
  const uint32_t sample_size = map_size * n_classes;

  float total_weight_val = 0;
  float output_val = 0;

  // parallel reduction buffer
  float* buffer = (float*)g_reduction_buffer;

  // here we assume weight is always 1
  total_weight_val = static_cast<float>(batch_size * map_size);

  for (size_t b = 0; b < batch_size; b++) {
    hb_tiled_for(map_size, [&](size_t elem) {
      const uint32_t cur_target = static_cast<uint32_t>(target_data[b * map_size + elem]);

      if (cur_target != ignore_index) {
        hb_assert(cur_target >= 0 && cur_target < n_classes);
        // for parallel reduction, we change -= to +=
        output_val += input_data[b * sample_size + cur_target * map_size + elem];
      }
    });
  }

  buffer[__bsg_id] = output_val;
  g_barrier.sync();

  if(__bsg_id == 0) {

    output_val = 0;
    for(size_t idx = 0; idx < bsg_tiles_X * bsg_tiles_Y; idx++) {
      output_val -= buffer[idx];
    }

    if (reduction == Reduction::Mean &&
        (total_weight_val != 0 || input.numel() == 0)) {
      // allow NaN result for total_weight_val == 0 case, see #15870
      output_val /= total_weight_val;
    }

    output(0) = output_val;
    total_weight(0) = total_weight_val;
  }

  g_barrier.sync();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib_nll_loss2d_forward, hb_tensor_t*,hb_tensor_t*,
                   hb_tensor_t*, hb_tensor_t*, uint32_t*, int32_t*)

}
