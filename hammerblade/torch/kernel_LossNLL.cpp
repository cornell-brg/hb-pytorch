//====================================================================
// LossNLL kernel
// 03/25/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <hb_reduction.hpp>

template<typename F>
static int tensorlib_lossnll_impl(
        hb_tensor_t* output_p,
        hb_tensor_t* total_weight_p,
        hb_tensor_t* input_p,
        hb_tensor_t* target_p,
        uint32_t*     reduction_p,
        int32_t*      ignore_index_p,
        F             weight) {

  HBTensor<float> output(output_p);
  HBTensor<float> total_weight(total_weight_p);
  HBTensor<float> input(input_p);
  HBTensor<int>   target(target_p);
  uint32_t reduction = *reduction_p;
  int32_t ignore_index = *ignore_index_p;

  const auto n_classes = input.dim(1);
  const auto batch_size = input.dim(0);
  const auto n_dims = input.ndim();

  if(reduction == Reduction::None && n_dims == 2) {
    hb_parallel_for(batch_size,
        [&](size_t i) {
          const auto cur_target = target(i);
          if (cur_target == ignore_index) {
            output(i) = 0;
          } else {
            hb_assert(cur_target >= 0 && cur_target < n_classes);
            const float cur_weight = weight(cur_target);
            output(i) = -input(i, cur_target) * cur_weight;
          }
        });

    return 0;
  }

  if (__bsg_id != 0) {
    return 0;
  }

  float output_val = 0;
  float total_weight_val = 0;

  if (n_dims == 1) {
    const auto cur_target = target(0);
    if (cur_target != ignore_index) {
      hb_assert(cur_target >= 0 && cur_target < n_classes);
      total_weight_val = weight(cur_target);
      output_val = -input(cur_target) * total_weight_val;
    }
  } else if (n_dims == 2) {
    hb_assert(target.dim(0) == batch_size);
    for (size_t i = 0; i < batch_size; i++) {
      const auto cur_target = target(i);

      if (cur_target != ignore_index) {
        hb_assert(cur_target >= 0 && cur_target < n_classes);
        const float cur_weight = weight(cur_target);
        total_weight_val += cur_weight;
        output_val -= input(i, cur_target) * cur_weight;
      }
    }
  }

  if (reduction == Reduction::Mean &&
      (total_weight_val != 0 || input.numel() == 0)) {
    output_val /= total_weight_val;
  }

  output(0) = output_val;
  total_weight(0) = total_weight_val;

  return 0;

}

extern "C" {


  __attribute__ ((noinline))  int tensorlib_lossnll_weight(
          hb_tensor_t* output_p,
          hb_tensor_t* total_weight_p,
          hb_tensor_t* input_p,
          hb_tensor_t* target_p,
          hb_tensor_t* weight_p,
          uint32_t*     reduction_p,
          int32_t*      ignore_index_p) {

    HBTensor<float> weight(weight_p);
    tensorlib_lossnll_impl(output_p,
                           total_weight_p,
                           input_p,
                           target_p,
                           reduction_p,
                           ignore_index_p,
                           [&](int i) {
                             return weight(i);
                           });

    return 0;

  }

  HB_EMUL_REG_KERNEL(tensorlib_lossnll_weight, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, uint32_t*, int32_t*)


  __attribute__ ((noinline))  int tensorlib_lossnll(
          hb_tensor_t* output_p,
          hb_tensor_t* total_weight_p,
          hb_tensor_t* input_p,
          hb_tensor_t* target_p,
          uint32_t*     reduction_p,
          int32_t*      ignore_index_p) {

    tensorlib_lossnll_impl(output_p,
                           total_weight_p,
                           input_p,
                           target_p,
                           reduction_p,
                           ignore_index_p,
                           [&](int i) {
                             return (float)1.0f;
                           });
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_lossnll, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, uint32_t*, int32_t*)

}
