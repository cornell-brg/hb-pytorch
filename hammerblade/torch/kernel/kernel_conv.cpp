//====================================================================
// Convolution kernel
// 03/08/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>
#include <kernel_conv.hpp>

static int convolution_forward(
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
  //
  // Width is equal to kernel buffer size because we load
  // input column by column. Height can be more to reduce
  // number of remote loads.
  register float X_local[XhBufSize][KwBufSize];

  // Number of outputs the data in X_local buffer can compute. This
  // only depneds on the height of the buffer as width is equal to
  // one kernel's width.
  uint32_t NumLocalOutputs = XhBufSize - Kh + 1;

  if(__bsg_id == 0)
    hb_assert_msg(Kh <= KhBufSize && Kw <= KwBufSize,
                  "Conv2d filter doesn't fit in DMEM allocated array");

  // Start profiling
  bsg_cuda_print_stat_kernel_start();

  for(uint32_t n = 0; n < N; ++n) {
    for(uint32_t ci = 0; ci < Cin; ++ci) { // input channel first to maximum data reuse
      hb_blocked_for(bsg_tiles_X * bsg_tiles_Y, Cout,
                    [&](size_t co, size_t tg_size_co) {
        // Load the filter w(co, ci, :, :) to dmem
        uint32_t w_offset = w.offset(co, ci, 0, 0);
        auto w_ptr = (__remote float*) w.data_ptr();
        load_weights(W_local, w_ptr, w_offset, Kh, Kw);

        size_t yh_start, yh_end, tg_size_yh;
        blocked_range(tg_size_co, Hout, yh_start, yh_end, tg_size_yh);

        for(size_t yh = yh_start; yh < yh_end; yh += NumLocalOutputs) {
          // Number of local outputs in this iteration
          uint32_t num_local_outputs = std::min((uint32_t) (yh_end - yh),
                                                NumLocalOutputs);

          hb_range yw_range;
          calc_range(&yw_range, Wout, tg_size_yh);
          size_t yw_start = yw_range.start;
          size_t yw_end   = yw_range.end;

          // width offset for the accessing local circular buffer
          uint32_t w_off = (Sw * yw_start) % Kw;

          uint32_t xhl_end = num_local_outputs + Kh -1;
          int32_t xh_start =  Sh * yh - Ph;

          // Load input to local buffer
          for(uint32_t xhl = 0; xhl < xhl_end; ++xhl) {
            uint32_t kw_local = w_off;
            uint32_t xh = xhl + xh_start;

            for(uint32_t kw = 0; kw < Kw; ++kw) {
              int32_t xw = Sw * yw_start - Pw + kw;

              if(xh >= 0 && xh < Hin && xw >= 0 && xw < Win)
                X_local[xhl][kw_local] = x(n, ci, xh, xw);
              else
                X_local[xhl][kw_local] = 0.0;

              kw_local = (kw_local == (Kw - 1)) ? 0 : kw_local + 1;
            }
          }

          for(uint32_t yw = yw_start; yw < yw_end; ++yw) {
            w_off = (Sw * yw) % Kw;

            if(yw != yw_start) {
              uint32_t kw_local = (Sw * (yw - 1)) % Kw;

              // Load a new column of input data
              for(uint32_t sw = 0; sw < Sw; ++sw) {
                int32_t xw = Sw * yw - Pw + Kw - Sw + sw;

                for(uint32_t xhl = 0; xhl < xhl_end; ++xhl) {
                  int32_t xh = xhl + xh_start;

                  X_local[xhl][kw_local] =
                      (xh >= 0 && xh < Hin && xw >= 0 && xw < Win) ?
                        x(n, ci, xh, xw) : 0;
                }

                kw_local = (kw_local == (Kw - 1)) ? 0 : kw_local + 1;
              }
            } // y == yw_start would be loaded in the outer loop

            for(uint32_t yh_local = 0; yh_local < num_local_outputs; ++yh_local) {
              for(uint32_t kh = 0; kh < Kh; ++kh) {
                uint32_t kw_local = w_off;
                uint32_t kh_local = yh_local + kh;
                uint32_t yh_remote = yh_local + yh;

                for(uint32_t kw = 0; kw < Kw; ++kw) {
                  if((ci + kh + kw) == 0) {
                    y(n, co, yh_remote, yw) = 0.0;
                  }

                  y(n, co, yh_remote, yw) += X_local[kh_local][kw_local] *
                                             W_local[kh][kw];

                  kw_local = (kw_local == (Kw - 1)) ? 0 : kw_local + 1;
                }
              }
            }
          }
        }
      });
    }
  };

  // End profiling
  bsg_cuda_print_stat_kernel_end();

  g_barrier.sync();
  return 0;
};

static int convolution_backward_input(
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

static int convolution_backward_weight(
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

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_convolution_forward(
          hb_tensor_t* output,
          hb_tensor_t* input,
          hb_tensor_t* weight,
          hb_vector_t* padding,
          hb_vector_t* strides) {
    #define CONV_FORWARD_TEMPLATED(                                        \
        N, Cout, Hout, Wout, Cin, Hin, Win, Kh, Kw, Sh, Sw, Ph, Pw)        \
      if(!convolution_forward_template<                                    \
              N, Cout, Hout, Wout, Cin, Hin, Win, Kh, Kw, Sh, Sw, Ph, Pw>( \
                  output, input, weight, padding, strides))                \
        return 0;

    const uint32_t N = 8;
    CONV_FORWARD_TEMPLATED(N,  16, 32, 32,  3, 32, 32, 3, 3, 1, 1, 1, 1);
    CONV_FORWARD_TEMPLATED(N,  32, 32, 32, 16, 32, 32, 3, 3, 1, 1, 1, 1);
    CONV_FORWARD_TEMPLATED(N,  32, 32, 32, 16, 32, 32, 1, 1, 1, 1, 0, 0);
    CONV_FORWARD_TEMPLATED(N,  64, 16, 16, 32, 16, 16, 3, 3, 1, 1, 1, 1);
    CONV_FORWARD_TEMPLATED(N,  64, 16, 16, 32, 16, 16, 1, 1, 1, 1, 0, 0);
    CONV_FORWARD_TEMPLATED(N, 128,  8,  8, 64,  8,  8, 3, 3, 1, 1, 1, 1);
    CONV_FORWARD_TEMPLATED(N, 128,  8,  8, 64,  8,  8, 1, 1, 1, 1, 0, 0);

    return convolution_forward(output, input, weight, padding, strides);
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
    #define CONV_BACKWARD_INPUT_TEMPLATED(                                 \
        N, Cout, Hout, Wout, Cin, Hin, Win, Kh, Kw, Sh, Sw, Ph, Pw)        \
      if(!convolution_backward_input_template<                             \
              N, Cout, Hout, Wout, Cin, Hin, Win, Kh, Kw, Sh, Sw, Ph, Pw>( \
                  grad_input, grad_output, weight, padding, strides))      \
        return 0;

    const uint32_t N = 8;
    CONV_BACKWARD_INPUT_TEMPLATED(N,  16, 32, 32,  3, 32, 32, 3, 3, 1, 1, 1, 1);
    CONV_BACKWARD_INPUT_TEMPLATED(N,  32, 32, 32, 16, 32, 32, 3, 3, 1, 1, 1, 1);
    CONV_BACKWARD_INPUT_TEMPLATED(N,  32, 32, 32, 16, 32, 32, 1, 1, 1, 1, 0, 0);
    CONV_BACKWARD_INPUT_TEMPLATED(N,  64, 16, 16, 32, 16, 16, 3, 3, 1, 1, 1, 1);
    CONV_BACKWARD_INPUT_TEMPLATED(N,  64, 16, 16, 32, 16, 16, 1, 1, 1, 1, 0, 0);
    CONV_BACKWARD_INPUT_TEMPLATED(N, 128,  8,  8, 64,  8,  8, 3, 3, 1, 1, 1, 1);
    CONV_BACKWARD_INPUT_TEMPLATED(N, 128,  8,  8, 64,  8,  8, 1, 1, 1, 1, 0, 0);

    return convolution_backward_input(grad_input, grad_output, weight, padding,
                                      strides);
  }

  __attribute__ ((noinline))  int tensorlib_convolution_backward_weight(
          hb_tensor_t* grad_weight,
          hb_tensor_t* grad_output,
          hb_tensor_t* input,
          hb_vector_t* padding,
          hb_vector_t* strides) {
    #define CONV_BACKWARD_WEIGHT_TEMPLATED(                                \
        N, Cout, Hout, Wout, Cin, Hin, Win, Kh, Kw, Sh, Sw, Ph, Pw)        \
      if(!convolution_backward_weight_template<                            \
              N, Cout, Hout, Wout, Cin, Hin, Win, Kh, Kw, Sh, Sw, Ph, Pw>( \
                  grad_weight, grad_output, input, padding, strides))      \
        return 0;

    const uint32_t N = 8;
    CONV_BACKWARD_WEIGHT_TEMPLATED(N,  16, 32, 32,  3, 32, 32, 3, 3, 1, 1, 1, 1);
    CONV_BACKWARD_WEIGHT_TEMPLATED(N,  32, 32, 32, 16, 32, 32, 3, 3, 1, 1, 1, 1);
    CONV_BACKWARD_WEIGHT_TEMPLATED(N,  32, 32, 32, 16, 32, 32, 1, 1, 1, 1, 0, 0);
    CONV_BACKWARD_WEIGHT_TEMPLATED(N,  64, 16, 16, 32, 16, 16, 3, 3, 1, 1, 1, 1);
    CONV_BACKWARD_WEIGHT_TEMPLATED(N,  64, 16, 16, 32, 16, 16, 1, 1, 1, 1, 0, 0);
    CONV_BACKWARD_WEIGHT_TEMPLATED(N, 128,  8,  8, 64,  8,  8, 3, 3, 1, 1, 1, 1);
    CONV_BACKWARD_WEIGHT_TEMPLATED(N, 128,  8,  8, 64,  8,  8, 1, 1, 1, 1, 0, 0);

    return convolution_backward_weight(grad_weight, grad_output, input, padding,
                                       strides);
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
