//====================================================================
// Convolution kernel
// 03/08/2020 Bandhav Veluri
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_convolution_forward(
          bsg_tensor_t* output,
          bsg_tensor_t* input,
          bsg_tensor_t* weight,
          bsg_vector_t* padding,
          bsg_vector_t* strides) {
    auto y = BSGTensor<float>(output);
    auto x = BSGTensor<float>(input);
    auto w = BSGTensor<float>(weight);
    auto p = BSGVector<uint32_t>(padding);
    auto s = BSGVector<uint32_t>(strides);

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
                    if((ci + kh + kw) == 0) {
                      y(n, co, yh, yw) = 0.0;
                    }

                    int32_t xh = Sh * yh - Ph + kh;
                    int32_t xw = Sw * yw - Pw + kw;

                    if(xh >= 0 && xh < Hin && xw >= 0 && xw < Win) {
                      y(n, co, yh, yw) += x(n, ci, xh, xw) * w(co, ci, kh, kw);
                    } // else 0
                  }
    }

    // End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  __attribute__ ((noinline))  int tensorlib_convolution_add_bias(
          bsg_tensor_t* output,
          bsg_tensor_t* bias) {
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
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_convolution_forward,
     bsg_tensor_t*, bsg_tensor_t*, bsg_tensor_t*,
     bsg_vector_t*, bsg_vector_t*);

  HB_EMUL_REG_KERNEL(tensorlib_convolution_add_bias,
     bsg_tensor_t*, bsg_tensor_t*);

}
