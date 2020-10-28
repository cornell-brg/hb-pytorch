//====================================================================
// Pooling kernel
// 03/19/2020 Bandhav Veluri
//====================================================================

#define KERNEL_DIM 2
#define STRIDE_X 2
#define STRIDE_Y 2
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

    hb_assert(KERNEL_DIM == Kh);
    hb_assert(KERNEL_DIM == Kw);
    hb_assert(STRIDE_Y == Sh);
    hb_assert(STRIDE_X == Sw);

    size_t num_blocks = Hout * C * N;

    bsg_attr_remote float* input_base = (float*)x.data_ptr();
    bsg_attr_remote float* output_base = (float*)y.data_ptr();
    bsg_attr_remote int* ind_base = (int*)ind.data_ptr();
    const uint32_t* input_strides = x.get_strides();
    const uint32_t* output_strides = y.get_strides();

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // main loop
    for (size_t idx = bsg_id; idx < num_blocks; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
      if (idx < num_blocks) {

        // calculate which output we need to produce -- there is no reuse so ...
        size_t image_id = idx / (Hout * C);
        size_t tmp = idx % (Hout * C);
        size_t channel_id = tmp / Hout;
        size_t yy = tmp % Hout;

        // calculate where to find the input data and where to write the output
        size_t index_offset = yy * STRIDE_Y * input_strides[2];
        size_t offset = image_id * input_strides[0] + channel_id * input_strides[1] + index_offset;
        size_t output_offset = image_id * output_strides[0] + channel_id * output_strides[1]
                               + yy * output_strides[2];

        bsg_attr_remote float* input_ptr = input_base + offset;
        bsg_attr_remote float* output_ptr = output_base + output_offset;
        bsg_attr_remote int* ind_ptr = ind_base + output_offset;

        bsg_unroll(6)
        for (size_t xx = 0; xx < Wout; xx++) {
          // fully unroll as we know kernel is 2x2
          float max = std::numeric_limits<float>::lowest();
          int index;
          register float tmp0 = *(input_ptr + 0);
          register float tmp1 = *(input_ptr + 1);
          register float tmp2 = *(input_ptr + input_strides[2] + 0);
          register float tmp3 = *(input_ptr + input_strides[2] + 1);
          asm volatile("": : :"memory");

          if (tmp0 > max) {
            max = tmp0;
            index = index_offset;
          }
          if (tmp1 > max) {
            max = tmp1;
            index = index_offset + 1;
          }
          if (tmp2 > max) {
            max = tmp2;
            index = index_offset + input_strides[2];
          }
          if (tmp3 > max) {
            max = tmp3;
            index = index_offset + input_strides[2] + 1;
          }

          *output_ptr = max;
          *ind_ptr = index;

          ind_ptr++;
          output_ptr++;
          input_ptr += Sw;
          index_offset += Sw;
        }
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

    hb_assert(KERNEL_DIM == Kh);
    hb_assert(KERNEL_DIM == Kw);
    hb_assert(STRIDE_Y == Sh);
    hb_assert(STRIDE_X == Sw);

    size_t num_blocks = Hout * C * N;

    bsg_attr_remote float* input_base = (float*)x.data_ptr();
    bsg_attr_remote float* output_base = (float*)y.data_ptr();
    bsg_attr_remote int* ind_base = (int*)ind.data_ptr();
    const uint32_t* input_strides = x.get_strides();
    const uint32_t* output_strides = y.get_strides();

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // main loop
    for (size_t idx = bsg_id; idx < num_blocks; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
      if (idx < num_blocks) {

        // calculate which output we need to produce -- each block is a row in grad_output
        size_t image_id = idx / (Hout * C);
        size_t tmp = idx % (Hout * C);
        size_t channel_id = tmp / Hout;
        size_t yy = tmp % Hout;

        // calculate where to find the input data and where to write the output
        size_t index_offset = yy * STRIDE_Y * input_strides[2];
        size_t offset = image_id * input_strides[0] + channel_id * input_strides[1] + index_offset;
        size_t output_offset = image_id * output_strides[0] + channel_id * output_strides[1]
                               + yy * output_strides[2];

        bsg_attr_remote float* input_ptr = input_base + offset;
        bsg_attr_remote float* output_ptr = output_base + output_offset;
        bsg_attr_remote int* ind_ptr = ind_base + output_offset;

        bsg_unroll(6)
        for (size_t xx = 0; xx < Wout; xx++) {
          int index = *ind_ptr;
          float grad = *output_ptr;

          // fully unroll as we know kernel is 2x2
          register float tmp0 = 0;
          register float tmp1 = 0;
          register float tmp2 = 0;
          register float tmp3 = 0;

          if (index == index_offset) {
            tmp0 = grad;
          } else if (index == index_offset + 1) {
            tmp1 = grad;
          } else if (index == index_offset + input_strides[2]) {
            tmp2 = grad;
          } else {
            tmp3 = grad;
          }

          *(input_ptr + 0) = tmp0;
          *(input_ptr + 1) = tmp1;
          *(input_ptr + input_strides[2] + 0) = tmp2;
          *(input_ptr + input_strides[2] + 1) = tmp3;

          ind_ptr++;
          output_ptr++;
          input_ptr += Sw;
          index_offset += Sw;
        }
      }
    }

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
