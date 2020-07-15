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
      hb_tensor_t* running_mean_, hb_tensor_t* running_var_, int* train_,
      float* eps_) {
    HBTensor<float, 4> output(output_);
    HBTensor<float, 4> input(input_);
    HBTensor<float, 1> weight(weight_);
    HBTensor<float, 1> bias(bias_);
    HBTensor<float, 1> save_mean(save_mean_);
    HBTensor<float, 1> save_invstd(save_invstd_);
    HBTensor<float, 1> running_mean(running_mean_);
    HBTensor<float, 1> running_var(running_var_);
    int train = *train_;
    float eps = *eps_;

    uint32_t N = input.get_sizes()[0];
    uint32_t C = input.get_sizes()[1];
    uint32_t H = input.get_sizes()[2];
    uint32_t W = input.get_sizes()[3];

    for(size_t c = 0; c < C; ++c) {
      float mean, invstd;
      if(train) {
        mean = save_mean(c);
        invstd = save_invstd(c);
      } else {
        mean = running_mean(c);
        invstd = 1 / sqrt(running_var(c) + eps);
      }

      float gamma = weight.numel() ? weight(c) : 1.0;
      float beta = bias.numel() ? bias(c) : 0.0;

      hb_tiled_for(N * H * W, [&](size_t i) {
          size_t w = i % W;
          size_t h = (i / W) % H;
          size_t n = (i / (W * H)) % N;

          output(n, c, h, w) = ((input(n, c, h, w) - mean) *
                                invstd) * gamma + beta;
      });
    }

    g_barrier.sync();
    return 0;
  }

  __attribute__ ((noinline))  int tensorlib_batch_norm2d_update_stats(
      hb_tensor_t* save_mean_, hb_tensor_t* save_invstd_, hb_tensor_t* input_, 
      hb_tensor_t* running_mean_, hb_tensor_t* running_var_, float* momentum_,
      float* eps_) {
    HBTensor<float, 1> save_mean(save_mean_);
    HBTensor<float, 1> save_invstd(save_invstd_);
    HBTensor<float, 4> input(input_);
    HBTensor<float, 1> running_mean(running_mean_);
    HBTensor<float, 1> running_var(running_var_);
    float momentum = *momentum_;
    float eps = *eps_;

    uint32_t N = input.get_sizes()[0];
    uint32_t C = input.get_sizes()[1];
    uint32_t H = input.get_sizes()[2];
    uint32_t W = input.get_sizes()[3];
    uint32_t numel = input.numel() / C;

    for(size_t c = 0; c < C; ++c) {
      float* reduction_buffer = (float*) g_reduction_buffer;

      // Compute partial sum and store it in
      // reduction buffer
      float partial_sum = 0.0;
      hb_tiled_for(N * H * W, [&](size_t i) {
          size_t w = i % W;
          size_t h = (i / W) % H;
          size_t n = (i / (W * H)) % N;

          partial_sum += input(n, c, h, w);
      });
      reduction_buffer[__bsg_id] = partial_sum;
      g_barrier.sync();

      // Use tile 0 to compute the mean
      if(__bsg_id == 0) {
        float sum = 0.0;
        for(size_t i = 0; i < bsg_tiles_X * bsg_tiles_Y; ++i) {
          sum += reduction_buffer[i];
        }
        save_mean(c) = sum / numel;
      }
      g_barrier.sync();

      // Compute partial sum and store it in
      // reduction buffer
      float partial_var_sum = 0.0;
      hb_tiled_for(N * H * W, [&](size_t i) {
          size_t w = i % W;
          size_t h = (i / W) % H;
          size_t n = (i / (W * H)) % N;

          partial_var_sum += (input(n, c, h, w) - save_mean(c)) *
                               (input(n, c, h, w) - save_mean(c));
      });
      reduction_buffer[__bsg_id] = partial_var_sum;
      g_barrier.sync();

      // Use tile 0 to compute remaining stats
      if(__bsg_id == 0) {
        float var_sum = 0.0;
        for(size_t i = 0; i < bsg_tiles_X * bsg_tiles_Y; ++i) {
          var_sum += reduction_buffer[i];
        }
        save_invstd(c) = 1 / sqrt((var_sum / numel) + eps);

        // These are optional arguments for this kernel, so check
        // for emptiness.
  
        if(running_mean.numel()) {
          running_mean(c) = momentum * save_mean(c) +
                              (1 - momentum) * running_mean(c);
        }
  
        if(running_var.numel()) {
          running_var(c) = momentum * (var_sum / (numel - 1)) +
                             (1 - momentum) * running_var(c);
        }
      } 
      g_barrier.sync();
    }

    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_batch_norm2d_transform_input,
      hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
      hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
      int*, float*);

  HB_EMUL_REG_KERNEL(tensorlib_batch_norm2d_update_stats,
      hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
      hb_tensor_t*, float*, float*);

}
