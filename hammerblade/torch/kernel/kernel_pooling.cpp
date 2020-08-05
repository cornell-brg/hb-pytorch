//====================================================================
// Pooling kernel
// 03/19/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>
#include <limits>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_max_pool2d(
          hb_tensor_t* output,
          hb_tensor_t* input,
          hb_tensor_t* indices,
          int* kH, int* kW,
          int* sH, int* sW,
          int* padH, int* padW,
          int* dilationH, int* dilationW) {
    auto y = HBTensor<float>(output);
    auto x = HBTensor<float>(input);
    auto ind = HBTensor<int>(indices);

    // max_pool2d parameters
    auto N = y.dim(0); // number of minibatches
    auto C = y.dim(1); // number of channels
    auto Hout = y.dim(2);
    auto Wout = y.dim(3);
    auto Hin = x.dim(2);
    auto Win = x.dim(3);
    auto Kh = *kH;
    auto Kw = *kW;
    auto Sh = *sH;
    auto Sw = *sW;
    auto Ph = *padH;
    auto Pw = *padW;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    hb_tiled_for(bsg_tiles_X * bsg_tiles_Y,
                 [&](size_t n, size_t c, size_t yh, size_t yw) {
      y(n, c, yh, yw) = std::numeric_limits<float>::lowest();

      for(uint32_t kh = 0; kh < Kh; ++kh)
        for(uint32_t kw = 0; kw < Kw; ++kw) {
          int32_t xh = Sh * yh - Ph + kh;
          int32_t xw = Sw * yw - Pw + kw;

          if(xh >= 0 && xh < Hin && xw >= 0 && xw < Win) {
            if(x(n, c, xh, xw) > y(n, c, yh, yw)) {
              y(n, c, yh, yw) = x(n, c, xh, xw);
              ind(n, c, yh, yw) = xh * Win + xw;
            }
          }
        }
    }, N, C, Hout, Wout);

    // End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  __attribute__ ((noinline))  int tensorlib_max_pool2d_backward(
          hb_tensor_t* gradInput,
          hb_tensor_t* gradOutput,
          hb_tensor_t* indices,
          hb_tensor_t* input,
          int* kH, int* kW,
          int* sH, int* sW,
          int* padH, int* padW) {
    auto y = HBTensor<float>(gradOutput);
    auto x = HBTensor<float>(gradInput);
    auto ind = HBTensor<int>(indices);

    // max_pool2d parameters
    auto N = y.dim(0); // number of minibatches
    auto C = y.dim(1); // number of channels
    auto Hout = y.dim(2);
    auto Wout = y.dim(3);
    auto Hin = x.dim(2);
    auto Win = x.dim(3);
    auto Kh = *kH;
    auto Kw = *kW;
    auto Sh = *sH;
    auto Sw = *sW;
    auto Ph = *padH;
    auto Pw = *padW;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(x, []() {return 0.0;});
    g_barrier.sync();

    hb_tiled_for(bsg_tiles_X * bsg_tiles_Y,
                 [&](size_t n, size_t c, size_t xh, size_t xw) {
      for(uint32_t kh = 0; kh < Kh; ++kh)
        for(uint32_t kw = 0; kw < Kw; ++kw) {
          uint32_t rel_h = xh - kh + Ph;
          uint32_t rel_w = xw - kw + Pw;

          if((rel_h % Sh != 0) || (rel_w % Sw != 0))
            continue;

          uint32_t yh = rel_h / Sh;
          uint32_t yw = rel_w / Sw;

          if(yh >= 0 && yh < Hout && yw >= 0 && yw < Wout
             && xh == ind(n, c, yh, yw) / Win
             && xw == ind(n, c, yh, yw) % Win) {
            x(n, c, xh, xw) += y(n, c, yh, yw);
          } // else 0
        }
    }, N, C, Hin, Win);

    // End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_max_pool2d,
    hb_tensor_t*,
    hb_tensor_t*,
    hb_tensor_t*,
    int*, int*,
    int*, int*,
    int*, int*,
    int*, int*);

  HB_EMUL_REG_KERNEL(tensorlib_max_pool2d_backward,
    hb_tensor_t*, hb_tensor_t*,
    hb_tensor_t*, hb_tensor_t*,
    int*, int*,
    int*, int*,
    int*, int*);

}
