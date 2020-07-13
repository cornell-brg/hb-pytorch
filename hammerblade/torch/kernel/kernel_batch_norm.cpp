//====================================================================
// BatchNorm kernel
// 07/11/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_batch_norm2d_transform_input(
      hb_tensor_t* output_, hb_tensor_t* input_, hb_tensor_t* weight_,
      hb_tensor_t* bias_, hb_tensor_t* save_mean_, hb_tensor_t* save_invstd_,
      hb_tensor_t* running_mean_, hb_tensor_t* running_var_, bool train,
      double eps) {
    HBTensor<float, 4> output(output_);
    HBTensor<float, 4> input(input_);
    HBTensor<float, 1> weight(weight_);
    HBTensor<float, 1> bias(bias_);
    HBTensor<float, 1> save_mean(save_mean_);
    HBTensor<float, 1> save_invstd(save_invstd_);
    HBTensor<float, 1> running_mean(running_mean_);
    HBTensor<float, 1> running_var(running_var_);

    if(__bsg_id == 0) {
      for(size_t c = 0; c < input.get_sizes()[1]; ++c) {
        float mean, invstd;
        if(train) {
          mean = save_mean(c);
          invstd = save_invstd(c);
        } else {
          mean = running_mean(c);
          invstd = 1 / sqrt(running_var(c) + eps);
        }

        float w = weight.numel() ? weight(c) : 1;
        float b = bias.numel() ? bias(c) : 0;

        for(size_t n = 0; n < input.get_sizes()[0]; ++n) { 
          for(size_t h = 0; h < input.get_sizes()[2]; ++h) {
            for(size_t w = 0; w < input.get_sizes()[3]; ++w) {
              output(n, c, h, w) = ((input(n, c, h, w) - mean) *
                                    invstd) * w + b;
            }
          }
        }
      }
    }

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_batch_norm2d_transform_input,
      hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
      hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
      bool, double);

}
