//====================================================================
// Convolution kernel
// 03/08/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>
#include "kernel_conv.hpp"

// Size of buffers allocated for filters in DMEM
const static uint32_t KhBufSize = 5;
const static uint32_t KwBufSize = 5;

inline void load_weights(float wl[KhBufSize][KwBufSize],
                         __remote float* NOALIAS wr,
                         uint32_t offset, uint32_t Kh, uint32_t Kw) {
  for(int i = 0; i < Kh; ++i) {
    for(int j = 0; j < Kw; ++j) {
      wl[i][j] = wr[offset + i * Kw + j];
    }
  }
}

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_convolution_forward(
          hb_tensor_t* output,
          hb_tensor_t* input,
          hb_tensor_t* weight,
          hb_vector_t* padding,
          hb_vector_t* strides) {
    auto y = HBTensor<float, 4>(output);
    auto x = HBTensor<float, 4>(input);
    auto w = HBTensor<float, 4>(weight);
    auto p = HBVector<uint32_t>(padding);
    auto s = HBVector<uint32_t>(strides);

    // Conv2d parameters
    auto N = y.dim(0); // number of minibatches
    auto Cout = y.dim(1); // number of output channels
    auto Hout = y.dim(2);
    auto Wout = y.dim(3);
    auto Cin = x.dim(1); // number of input channels
    auto Hin = x.dim(2);
    auto Win = x.dim(3);
    auto Kh = w.dim(2);
    auto Kw = w.dim(3);
    auto Sh = s[0];
    auto Sw = s[1];
    auto Ph = p[0];
    auto Pw = p[1];

    // Weights buffer
    register float W_local[KhBufSize][KwBufSize];

    // Circular buffer to hold inputs
    register float X_local[KhBufSize][KwBufSize];

    if(__bsg_id == 0)
      hb_assert_msg(Kh <= KhBufSize && Kw <= KwBufSize,
                    "Conv2d filter doesn't fit in DMEM allocated array");

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    for(uint32_t n = 0; n < N; ++n) {
      for(uint32_t ci = 0; ci < Cin; ++ci) { // input channel first to maximum data reuse
        blocked_for(bsg_tiles_X * bsg_tiles_Y, Cout,
                    [&](size_t co, size_t tg_size_co) {
          // Load the filter w(co, ci, :, :) to dmem
          uint32_t w_offset = w.offset(co, ci, 0, 0);
          auto w_ptr = (__remote float*) w.data_ptr();
          load_weights(W_local, w_ptr, w_offset, Kh, Kw);

          blocked_for(tg_size_co, Hout, [&](size_t yh, size_t tg_size_yh) {
            hb_tiled_for(tg_size_yh, Wout, [&](size_t yw) {
              for(uint32_t kh = 0; kh < Kh; ++kh) {
                for(uint32_t kw = 0; kw < Kw; ++kw) {
                  if((ci + kh + kw) == 0) {
                    y(n, co, yh, yw) = 0.0;
                  }

                  int32_t xh = Sh * yh - Ph + kh;
                  int32_t xw = Sw * yw - Pw + kw;

                  if(xh >= 0 && xh < Hin && xw >= 0 && xw < Win) {
                    y(n, co, yh, yw) += x(n, ci, xh, xw) * W_local[kh][kw];
                  } // else 0
                }
              }
            });
          });
        });
      }
    };

    // End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  };

  __attribute__ ((noinline))  int tensorlib_convolution_add_bias(
          hb_tensor_t* output,
          hb_tensor_t* bias) {
    auto y = (float*) ((intptr_t) output->data);
    auto y_strides = (uint32_t*) ((intptr_t) output->strides);
    auto b = (float*) ((intptr_t) bias->data);

    auto N = output->N;                // total number of elements in the output
    auto out_ch_stride = y_strides[1]; // output channel stride
    auto nb = bias->N;                 // number of elements in the bias

    // Calculate elements per tile
    uint32_t len_per_tile = N / (bsg_tiles_X * bsg_tiles_Y) + 1;
    uint32_t start = len_per_tile * __bsg_id;
    uint32_t end = start + len_per_tile;
    end = (end > N)  ? N : end;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    for(int i = start; i < end; ++i) {
      float bias = b[(i / out_ch_stride) % nb];
      y[i] += bias;
    }

    // End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  __attribute__ ((noinline))  int tensorlib_convolution_backward_input(
          hb_tensor_t* grad_input,
          hb_tensor_t* grad_output,
          hb_tensor_t* weight,
          hb_vector_t* padding,
          hb_vector_t* strides) {
    auto x = HBTensor<float>(grad_input);
    auto y = HBTensor<float>(grad_output);
    auto w = HBTensor<float>(weight);
    auto p = HBVector<uint32_t>(padding);
    auto s = HBVector<uint32_t>(strides);

    // Conv2d parameters
    auto N = y.dim(0); // number of minibatches
    auto Cout = y.dim(1); // number of output channels
    auto Hout = y.dim(2);
    auto Wout = y.dim(3);
    auto Cin = x.dim(1); // number of input channels
    auto Hin = x.dim(2);
    auto Win = x.dim(3);
    auto Kh = w.dim(2);
    auto Kw = w.dim(3);
    auto Sh = s[0];
    auto Sw = s[1];
    auto Ph = p[0];
    auto Pw = p[1];

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // init input grads
    hb_tiled_foreach([]() {return 0.0;}, x);
    g_barrier.sync();

    for(uint32_t n = 0; n < N; ++n)
    for(uint32_t co = 0; co < Cout; ++co)
      hb_tiled_for(bsg_tiles_X * bsg_tiles_Y,
                   [&](size_t ci, size_t xh, size_t xw) {
        for(uint32_t kh = 0; kh < Kh; ++kh)
          for(uint32_t kw = 0; kw < Kw; ++kw) {
            uint32_t rel_h = xh - kh + Ph;
            uint32_t rel_w = xw - kw + Pw;

            if((rel_h % Sh != 0) || (rel_w % Sw != 0))
              continue;

            uint32_t yh = rel_h / Sh;
            uint32_t yw = rel_w / Sw;

            if(yh >= 0 && yh < Hout && yw >= 0 && yw < Wout) {
              x(n, ci, xh, xw) += y(n, co, yh, yw) * w(co, ci, kh, kw);
            } // else 0
          }
      }, Cin, Hin, Win);

    // End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  __attribute__ ((noinline))  int tensorlib_convolution_backward_weight(
          hb_tensor_t* grad_weight,
          hb_tensor_t* grad_output,
          hb_tensor_t* input,
          hb_vector_t* padding,
          hb_vector_t* strides) {
    auto x = HBTensor<float>(input);
    auto y = HBTensor<float>(grad_output);
    auto w = HBTensor<float>(grad_weight);
    auto p = HBVector<uint32_t>(padding);
    auto s = HBVector<uint32_t>(strides);

    // Conv2d parameters
    auto N = y.dim(0); // number of minibatches
    auto Cout = y.dim(1); // number of output channels
    auto Hout = y.dim(2);
    auto Wout = y.dim(3);
    auto Cin = x.dim(1); // number of input channels
    auto Hin = x.dim(2);
    auto Win = x.dim(3);
    auto Kh = w.dim(2);
    auto Kw = w.dim(3);
    auto Sh = s[0];
    auto Sw = s[1];
    auto Ph = p[0];
    auto Pw = p[1];

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // init weight grads
    hb_tiled_foreach([]() {return 0.0;}, w);
    g_barrier.sync();

    for(uint32_t n = 0; n < N; ++n)
      hb_tiled_for(bsg_tiles_X * bsg_tiles_Y,
                   [&](size_t co, size_t ci, size_t kh, size_t kw) {
        for(uint32_t yh = 0; yh < Hout; ++yh)
          for(uint32_t yw = 0; yw < Wout; ++yw){
            int32_t xh = Sh * yh - Ph + kh;
            int32_t xw = Sw * yw - Pw + kw;

            if(xh >= 0 && xh < Hin && xw >= 0 && xw < Win) {
              w(co, ci, kh, kw) += y(n, co, yh, yw) * x(n, ci, xh, xw);
            } // else 0
          }
      }, Cout, Cin, Kh, Kw);

    // End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  __attribute__ ((noinline))  int tensorlib_convolution_backward_bias(
          hb_tensor_t* grad_bias,
          hb_tensor_t* grad_output) {
    auto gb = HBTensor<float>(grad_bias);
    auto y = HBTensor<float>(grad_output);

    auto N = y.dim(0); // number of minibatches
    auto Cout = y.dim(1); // number of output channels
    auto Hout = y.dim(2);
    auto Wout = y.dim(3);

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    hb_tiled_for(Cout, [&](size_t co) {
      for(uint32_t n = 0; n < N; ++n)
        for(uint32_t yh = 0; yh < Hout; ++yh)
          for(uint32_t yw = 0; yw < Wout; ++yw) {
            if((n + yh + yw) == 0)
              gb(co) = 0.0f;

            gb(co) += y(n, co, yh, yw);
          }
    });

    // End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_convolution_forward,
     hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
     hb_vector_t*, hb_vector_t*);

  HB_EMUL_REG_KERNEL(tensorlib_convolution_add_bias,
     hb_tensor_t*, hb_tensor_t*);

  HB_EMUL_REG_KERNEL(tensorlib_convolution_backward_input,
     hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
     hb_vector_t*, hb_vector_t*);

  HB_EMUL_REG_KERNEL(tensorlib_convolution_backward_weight,
     hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
     hb_vector_t*, hb_vector_t*);

  HB_EMUL_REG_KERNEL(tensorlib_convolution_backward_bias,
     hb_tensor_t*, hb_tensor_t*);

}
