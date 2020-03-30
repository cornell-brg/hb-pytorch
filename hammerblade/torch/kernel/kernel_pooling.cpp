//====================================================================
// Pooling kernel
// 03/19/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>
#include <float.h>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_max_pool2d(
          bsg_tensor_t* output,
          bsg_tensor_t* input,
          bsg_tensor_t* indices,
          int* kH, int* kW,
          int* sH, int* sW,
          int* padH, int* padW,
          int* dilationH, int* dilationW) {
    auto y = BSGTensor<float>(output);
    auto x = BSGTensor<float>(input);
    auto ind = BSGTensor<int>(indices);

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

    // Preliminary single tile implementation
    //
    // Grows O(^4) with image size:
    //   N x C x H x W
    //   Kernel loops are constant-time
    if(__bsg_id == 0) {
      for(uint32_t n = 0; n < N; ++n)
        for(uint32_t c = 0; c < C; ++c)
          for(uint32_t yh = 0; yh < Hout; ++yh)
            for(uint32_t yw = 0; yw < Wout; ++yw)
              for(uint32_t kh = 0; kh < Kh; ++kh)
                for(uint32_t kw = 0; kw < Kw; ++kw) {
                  if((kh + kw) == 0) {
                    y(n, c, yh, yw) = FLT_MIN;
                  }

                  int32_t xh = Sh * yh - Ph + kh;
                  int32_t xw = Sw * yw - Pw + kw;

                  if(xh >= 0 && xh < Hin && xw >= 0 && xw < Win) {
                    if(x(n, c, xh, xw) > y(n, c, yh, yw)) {
                      y(n, c, yh, yw) = x(n, c, xh, xw);
                      ind(n, c, yh, yw) = xh * Win + xw;
                    }
                  }
                }
    }

    // End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  __attribute__ ((noinline))  int tensorlib_max_pool2d_backward(
          bsg_tensor_t* gradInput,
          bsg_tensor_t* gradOutput,
          bsg_tensor_t* indices,
          bsg_tensor_t* input,
          int* kH, int* kW,
          int* sH, int* sW,
          int* padH, int* padW) {
    auto y = BSGTensor<float>(gradOutput);
    auto x = BSGTensor<float>(gradInput);
    auto ind = BSGTensor<int>(indices);

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

    if(__bsg_id == 0) {
      for(uint32_t n = 0; n < N; ++n)
        for(uint32_t c = 0; c < C; ++c)
          for(uint32_t xh = 0; xh < Hin; ++xh)
            for(uint32_t xw = 0; xw < Win; ++xw)
              x(n, c, xh, xw) = 0.0f;

      for(uint32_t n = 0; n < N; ++n)
        for(uint32_t c = 0; c < C; ++c)
          for(uint32_t yh = 0; yh < Hout; ++yh)
            for(uint32_t yw = 0; yw < Wout; ++yw) {
              int32_t xh = ind(n, c, yh, yw) / Win;
              int32_t xw = ind(n, c, yh, yw) % Win;

              x(n, c, xh, xw) += y(n, c, yh, yw);
            }
    }

    // End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_max_pool2d,
    bsg_tensor_t*,
    bsg_tensor_t*,
    bsg_tensor_t*,
    int*, int*,
    int*, int*,
    int*, int*,
    int*, int*);

  HB_EMUL_REG_KERNEL(tensorlib_max_pool2d_backward,
    bsg_tensor_t*, bsg_tensor_t*,
    bsg_tensor_t*, bsg_tensor_t*,
    int*, int*,
    int*, int*,
    int*, int*);

}
