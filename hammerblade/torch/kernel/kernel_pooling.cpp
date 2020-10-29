//====================================================================
// Pooling kernel
// 03/19/2020 Bandhav Veluri
//====================================================================

#define KERNEL_DIM 3
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

        bsg_unroll(3)
        for (size_t xx = 0; xx < Wout; xx++) {
          // fully unroll as we know kernel is 2x2 -- 3x3
          float max = std::numeric_limits<float>::lowest();
          int index;
          register float tmp0 = *(input_ptr + 0);
          register float tmp1 = *(input_ptr + 1);
          register float tmp2 = *(input_ptr + 2);
          register float tmp3 = *(input_ptr + input_strides[2] + 0);
          register float tmp4 = *(input_ptr + input_strides[2] + 1);
          register float tmp5 = *(input_ptr + input_strides[2] + 2);
          register float tmp6 = *(input_ptr + 2 * input_strides[2] + 0);
          register float tmp7 = *(input_ptr + 2 * input_strides[2] + 1);
          register float tmp8 = *(input_ptr + 2 * input_strides[2] + 2);
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
            index = index_offset + 2;
          }
          if (tmp3 > max) {
            max = tmp3;
            index = index_offset + input_strides[2] + 0;
          }
          if (tmp4 > max) {
            max = tmp4;
            index = index_offset + input_strides[2] + 1;
          }
          if (tmp5 > max) {
            max = tmp5;
            index = index_offset + input_strides[2] + 2;
          }
          if (tmp6 > max) {
            max = tmp6;
            index = index_offset + 2 * input_strides[2] + 0;
          }
          if (tmp7 > max) {
            max = tmp7;
            index = index_offset + 2 * input_strides[2] + 1;
          }
          if (tmp8 > max) {
            max = tmp8;
            index = index_offset + 2 * input_strides[2] + 2;
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

    // parallel over grad_in, which is the result here
    size_t num_blocks = Hin * C * N;

    bsg_attr_remote float* result_base = (float*)x.data_ptr();
    bsg_attr_remote float* grad_base = (float*)y.data_ptr();
    bsg_attr_remote int* ind_base = (int*)ind.data_ptr();
    const uint32_t* input_strides = x.get_strides();
    const uint32_t* grad_strides = y.get_strides();

    // local buffer
    float row_buffer[128];

    auto reset_buffer = [&]() {
      bsg_unroll(16)
      for (size_t idx = 0; idx < Win; idx++) {
        row_buffer[idx] = 0;
      }
    };

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // main loop
    for (size_t idx = bsg_id; idx < num_blocks; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
      if (idx < num_blocks) {

        // reset row buffer
        reset_buffer();

        // calculate which block we need to produce -- each block is a row in grad_input
        size_t image_id = idx / (Hin * C);
        size_t tmp = idx % (Hin * C);
        size_t channel_id = tmp / Hin;
        size_t yy = tmp % Hin;

        // first kernel that can cover my row
        size_t k_id = yy / STRIDE_Y;
        size_t k_row = k_id * STRIDE_Y;
        size_t k_read = 1;
        while (k_row > 0) {
          size_t tmp = k_row - STRIDE_Y;
          if (tmp + KERNEL_DIM > yy) {
            k_read++;
            k_row -= STRIDE_Y;
          } else {
            break;
          }
        }
        k_id = k_row / STRIDE_Y;

        // std::cout << "I'm row " << yy << " first kernel that covers me start at row " << k_row << " and there are " << k_read << " kernels that cover me" << std::endl;


        size_t grad_offset   = image_id * grad_strides[0] + channel_id * grad_strides[1] + k_id * grad_strides[2];
        size_t result_offset = image_id * input_strides[0] + channel_id * input_strides[1] + yy * input_strides[2];

        // which indices are we in charge of?
        size_t row_idx_start = yy * Win;
        size_t row_idx_end   = row_idx_start + Win;

        bsg_attr_remote float* grad_ptr   = grad_base + grad_offset;
        bsg_attr_remote float* result_ptr = result_base + result_offset;
        bsg_attr_remote int*    ind_ptr   = ind_base + grad_offset;

        // read all related grad rows and populate result
        for (size_t kk = 0; kk < k_read; kk++) {
          for (size_t xx = 0; xx < Wout; xx++) {
            int index = *(ind_ptr + xx);
            float grad = *(grad_ptr + xx);
            if (index >= row_idx_start && index < row_idx_end) {
              row_buffer[index - row_idx_start] += grad;
            }
          }
          grad_ptr += grad_strides[2];
          ind_ptr  += grad_strides[2];
        }

        // write back
        for (size_t xx = 0; xx < Win; xx++) {
          result_ptr[xx] = row_buffer[xx];
        }

        /*
        // calculate where to find the input data and where to write the output
        // size_t index_offset = yy * STRIDE_Y * input_strides[2];
        // size_t offset = image_id * input_strides[0] + channel_id * input_strides[1] + index_offset;
        // size_t output_offset = image_id * output_strides[0] + channel_id * output_strides[1]
        //                        + yy * output_strides[2];

        // bsg_attr_remote float* input_ptr = input_base + offset;
        // bsg_attr_remote float* output_ptr = output_base + output_offset;
        // bsg_attr_remote int* ind_ptr = ind_base + output_offset;

        bsg_unroll(6)
        for (size_t xx = 0; xx < Wout; xx++) {

          int index = *ind_ptr;
          float grad = *output_ptr;

          // fully unroll as we know kernel is 2x2 -- 3x3
          register float tmp0 = 0;
          register float tmp1 = 0;
          register float tmp2 = 0;
          register float tmp3 = 0;
          register float tmp4 = 0;
          register float tmp5 = 0;
          register float tmp6 = 0;
          register float tmp7 = 0;
          register float tmp8 = 0;

          if (index == index_offset) {
            tmp0 += grad;
          } else if (index == index_offset + 1) {
            tmp1 += grad;
          } else if (index == index_offset + 2) {
            tmp2 += grad;
          } else if (index == index_offset + input_strides[2] + 0) {
            tmp3 += grad;
          } else if (index == index_offset + input_strides[2] + 1) {
            tmp4 += grad;
          } else if (index == index_offset + input_strides[2] + 2) {
            tmp5 += grad;
          } else if (index == index_offset + 2 * input_strides[2] + 0) {
            tmp6 += grad;
          } else if (index == index_offset + 2 * input_strides[2] + 1) {
            tmp7 += grad;
          } else {
            tmp8 += grad;
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
        */
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
