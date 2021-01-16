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

    hb_assert(Kh == 1);
    hb_assert(Kw == 2);
    hb_assert(Kh == Sh);
    hb_assert(Kw == Sw);
    hb_assert(Ph == 0);
    hb_assert(Pw == 0);
    hb_assert(Hout == 1);

    bsg_attr_remote float* input_base = (float*)x.data_ptr();
    bsg_attr_remote float* output_base = (float*)y.data_ptr();
    bsg_attr_remote int* ind_base = (int*)ind.data_ptr();
    const uint32_t* input_strides = x.get_strides();
    const uint32_t* output_strides = y.get_strides();

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    for(size_t image_id = 0; image_id < N; image_id++) {
      for(size_t channel_id = 0; channel_id < C; channel_id++) {
        size_t output_offset = image_id * output_strides[0] + channel_id * output_strides[1];
        size_t input_offset  = image_id * input_strides[0] + channel_id * input_strides[1];
        hb_tiled_for(Wout, [&](size_t yy) {
          size_t output_offset2 = output_offset + yy * output_strides[3];
          size_t input_offset2  = input_offset  + yy * 2 * input_strides[3];
          bsg_attr_remote float* input_ptr  = input_base + input_offset2;
          bsg_attr_remote float* output_ptr = output_base + output_offset2;
          bsg_attr_remote int*   ind_ptr    = ind_base + output_offset2;

          float max;
          int index;
          register float tmp0 = *(input_ptr + 0);
          register float tmp1 = *(input_ptr + 1);
          if (tmp0 > tmp1) {
            max = tmp0;
            index = 0;
          } else {
            max = tmp1;
            index = 1;
          }
          *output_ptr = max;
          *ind_ptr = index;
        });
      }
    }

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

    hb_tiled_foreach([]() {return 0.0;}, x);
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
