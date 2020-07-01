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

    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    size_t end = Cout;
    int32_t local_mem = 300;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    float temp[1];
 
    for(uint32_t co = start; co < end; co = co + thread_num) {
      int32_t wrow[2];
      wrow[0] = w_row(co);
      wrow[1] = w_row(co+1);
      size_t nnz = wrow[1] - wrow[0];
      int32_t iter = 1;
      int32_t col_size = nnz;
      if(nnz > local_mem) {
        iter = std::ceil((float)(nnz) / (float)(local_mem));
        col_size = local_mem;
      }
      for(int32_t it = 0; it < iter; ++it) {
        int32_t col_start = it * col_size;
        int32_t col_end = (col_start + col_size) > nnz ? nnz : (col_start + col_size);
        int32_t length = col_end - col_start;
        int32_t wcol[length];
        float wval[length];
        int offset = wrow[0] + it * col_size;
        for(uint32_t ind = 0; ind < length; ++ind) {
          wcol[ind] = w_col(ind + offset);
          wval[ind] = w_val(ind + offset);
        }
        //bsg_printf("Row %d is performing on from %dth nnz to %dth nnz in iteration %d.\n", co, col_start, col_end, it);
        for(uint32_t n = 0; n < N; ++n) {
          for(uint32_t yh = 0; yh < Hout; ++yh) {
            for(uint32_t yw = 0; yw < Wout; ++yw) {
              temp[0] = y(n, co, yh, yw);
              int32_t xhoffset = Sh * yh - Ph;
              int32_t xwoffset = Sw * yw - Pw;
              for(uint32_t i = 0; i < length; i++) {
                // bsg_printf("The index of column is %d \n", index);
                int32_t w_1d = wcol[i];
                int32_t ci = w_1d / (Kh * Kw);
                int32_t rest = w_1d - ci * Kh * Kw;
                int32_t kh = rest / Kh;
                int32_t kw = rest - kh * Kh;

                int32_t xh = xhoffset + kh;
                int32_t xw = xwoffset + kw;

                if(xh >= 0 && xh < Hin && xw >= 0 && xw < Win) {
                  temp[0] += x(n, ci, xh, xw) * wval[i];
                  // bsg_printf("input is x[%d %d %d %d]\n", n, ci, xh, xw);
                } // else 0
              }
              y(n, co, yh, yw) = temp[0];
              //bsg_printf("output is y[%d %d %d %d]\n", n, co, yh, yw);
            }
          }
        }
      }
    }

    // End profiling
    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_sparse_convolution_forward, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
hb_vector_t*, hb_vector_t*, hb_vector_t*, hb_vector_t*)

}
