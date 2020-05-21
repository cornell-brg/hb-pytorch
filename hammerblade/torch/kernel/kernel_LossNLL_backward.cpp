//====================================================================
// LossNLL backward kernel
// 03/26/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <hb_reduction.hpp>

template<typename F>
static int tensorlib_lossnll_backward_impl(
       hb_tensor_t* grad_input_p,
       hb_tensor_t* grad_output_p,
       hb_tensor_t* input_p,
       hb_tensor_t* target_p,
       hb_tensor_t* total_weight_p,
       uint32_t*     reduction_p,
       int32_t*      ignore_index_p,
       F             weight) {

  HBTensor<float> grad_input(grad_input_p);
  HBTensor<float> grad_output(grad_output_p);
  HBTensor<float> input(input_p);
  HBTensor<int>   target(target_p);
  HBTensor<float> total_weight(total_weight_p);
  uint32_t reduction = *reduction_p;
  int32_t ignore_index = *ignore_index_p;

  const auto n_classes = input.dim(1);
  const auto batch_size = input.dim(0);
  const auto n_dims = input.ndim();

  if(reduction == Reduction::None && n_dims == 2) {
    // check_dim_size(grad_output, 1, 0, batch_size);
    hb_assert(grad_output.ndim() == 1 && grad_output.dim(0) == batch_size);
    hb_parallel_for(batch_size,
        [&](size_t i) {
          const auto cur_target = target(i);
          if (cur_target != ignore_index) {
            const float cur_weight = weight(cur_target);
            grad_input(i, cur_target) = -cur_weight * grad_output(i);
          }
        });
    return 0;
  }

  const float total_weight_value = total_weight(0);
  if (total_weight_value <= 0) {
    return 0;
  }

  hb_assert(grad_output.ndim() <= 1 && grad_output.numel() == 1);
  const float grad_output_value = grad_output(0);

  if (n_dims == 1 && __bsg_id == 0) {
    const auto cur_target = target(0);
    if (cur_target != ignore_index) {
      hb_assert(cur_target >= 0 && cur_target < n_classes);
      grad_input(cur_target) = -weight(cur_target);
      grad_input(cur_target) *= grad_output_value;
    }
  } else if (n_dims == 2) {
    hb_assert(target.dim(0) == batch_size);
    hb_parallel_for(batch_size,
        [&](size_t i) {
          const auto cur_target = target(i);

          if (cur_target != ignore_index) {
            hb_assert(cur_target >= 0 && cur_target < n_classes);
            const float cur_weight = weight(cur_target);
            grad_input(i, cur_target) = -cur_weight * grad_output_value;

            if (reduction == Reduction::Mean) {
              grad_input(i, cur_target) /= total_weight_value;
            }
          }
        });
  }

  return 0;

}

extern "C" {

  __attribute__ ((noinline))  int tensorlib_lossnll_backward_weight(
          hb_tensor_t* grad_input_p,
          hb_tensor_t* grad_output_p,
          hb_tensor_t* input_p,
          hb_tensor_t* target_p,
          hb_tensor_t* weight_p,
          hb_tensor_t* total_weight_p,
          uint32_t*     reduction_p,
          int32_t*      ignore_index_p) {

    HBTensor<float> weight(weight_p);
    tensorlib_lossnll_backward_impl(grad_input_p,
                                    grad_output_p,
                                    input_p,
                                    target_p,
                                    total_weight_p,
                                    reduction_p,
                                    ignore_index_p,
                                    [&](int i) {
                                      return weight(i);
                                    });

    g_barrier.sync();
    return 0;

  }

  HB_EMUL_REG_KERNEL(tensorlib_lossnll_backward_weight, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, uint32_t*,
                     int32_t*);

  __attribute__ ((noinline))  int tensorlib_lossnll_backward(
          hb_tensor_t* grad_input_p,
          hb_tensor_t* grad_output_p,
          hb_tensor_t* input_p,
          hb_tensor_t* target_p,
          hb_tensor_t* total_weight_p,
          uint32_t*     reduction_p,
          int32_t*      ignore_index_p) {

    tensorlib_lossnll_backward_impl(grad_input_p,
                                    grad_output_p,
                                    input_p,
                                    target_p,
                                    total_weight_p,
                                    reduction_p,
                                    ignore_index_p,
                                    [&](int i) {
                                      return (float)1.0f;
                                    });

    g_barrier.sync();
    return 0;

  }

  HB_EMUL_REG_KERNEL(tensorlib_lossnll_backward, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, uint32_t*, int32_t*);

}
