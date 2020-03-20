//====================================================================
// Pooling kernel
// 03/19/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>

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

    // Conv2d parameters
    auto N = y.dim(0); // number of minibatches
    auto Cout = y.dim(1); // number of output channels
    auto Hout = y.dim(2);
    auto Wout = y.dim(3);
    auto Cin = x.dim(1); // number of input channels
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
    // Grows O(^5) with image size:
    //   N x Cout x Cin x H x W
    //   Kernel loops are constant-time
    if(__bsg_id == 0) {
      for(uint32_t n = 0; n < N; ++n)
        for(uint32_t co = 0; co < Cout; ++co)
          for(uint32_t yh = 0; yh < Hout; ++yh)
            for(uint32_t yw = 0; yw < Wout; ++yw)
              for(uint32_t ci = 0; ci < Cin; ++ci)
                for(uint32_t kh = 0; kh < Kh; ++kh)
                  for(uint32_t kw = 0; kw < Kw; ++kw) {
                    int32_t xh = Sh * yh - Ph + kh;
                    int32_t xw = Sw * yw - Pw + kw;

                    if((ci + kh + kw) == 0) {
                      y(n, co, yh, yw) = x(n, ci, xh, xw);
                    } else if(xh >= 0 && xh < Hin && xw >= 0 && xw < Win) {
                      if(x(n, ci, xh, xw) > y(n, co, yh, yw)) {
                        y(n, co, yh, yw) = x(n, ci, xh, xw);
                      }
                    }
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

}
