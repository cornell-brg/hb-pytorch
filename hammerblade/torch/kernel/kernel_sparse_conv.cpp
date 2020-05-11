//====================================================================
// Sparse convolution kernel
// 05/08/2020 Zhongyuan Zhao
//====================================================================

#include <kernel_common.hpp>
#include <cmath>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_sparse_convolution_forward(
          hb_tensor_t* output,
          hb_tensor_t* input,
          hb_tensor_t* csr,
          hb_tensor_t* colindices,
          hb_tensor_t* values,
          hb_vector_t* padding,
          hb_vector_t* strides,
          hb_vector_t* input_sizes,
          hb_vector_t* weight_sizes) {

    auto y = HBTensor<float>(output);
    auto x = HBTensor<float>(input);
    auto w_row = HBTensor<int>(csr);
    auto w_col = HBTensor<int>(colindices);
    auto w_val = HBTensor<float>(values);
    
    auto p = HBVector<uint32_t>(padding);
    auto s = HBVector<uint32_t>(strides);
    auto in_dims = HBVector<uint32_t>(input_sizes);
    auto w_dims = HBVector<uint32_t>(weight_sizes);

    // Conv2d parameters
    auto N = y.dim(0); // number of minibatches
    auto Cout = y.dim(1); // number of output channels
    auto Hout = y.dim(2);
    auto Wout = y.dim(3);
    auto Cin = x.dim(1); // number of input channels
    auto Hin = x.dim(2);
    auto Win = x.dim(3);
    auto Kh = w_dims[2];
    auto Kw = w_dims[3];
    auto Sh = s[0];
    auto Sw = s[1];
    auto Ph = p[0];
    auto Pw = p[1];

    size_t len_per_tile = Cout  / (bsg_tiles_X * bsg_tiles_Y) + 1;
    size_t start = len_per_tile * __bsg_id;
    size_t end = start + len_per_tile;
    end = (end > Cout) ? Cout : end;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

      for(uint32_t n = 0; n < N; ++n) {
        for(uint32_t co = start; co < end; ++co) {
          for(uint32_t i = w_row(co); i < w_row(co+1); i++) {
            uint32_t w_1d = w_col(i);
            uint32_t ci = std::floor(w_1d / (Kh * Kw));
            uint32_t rest = w_1d - ci * Kh * Kw;
            uint32_t kh = std::floor(rest / Kh);
            uint32_t kw = rest - kh * Kh;
            float val = w_val(i);
            for(uint32_t yh = 0; yh < Hout; ++yh) {
              for(uint32_t yw = 0; yw < Wout; ++yw) {
                if((ci + kh + kw) == 0) {
                  y(n, co, yh, yw) = 0.0;
                }

                int32_t xh = Sh * yh - Ph + kh;
                int32_t xw = Sw * yw - Pw + kw;

                if(xh >= 0 && xh < Hin && xw >= 0 && xw < Win) {
                  y(n, co, yh, yw) += x(n, ci, xh, xw) * w_val(i);
                } // else 0
              }
            }
          }
        }
      }

    // End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sparse_convolution_forward, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
hb_vector_t*, hb_vector_t*, hb_vector_t*, hb_vector_t*)

}
