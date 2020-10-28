//====================================================================
// SPMD 2D Convolution
// Idea is that each tile will receive a piece of output image that
// does not overlap with any other tile to work on
// 10/02/2020 Lin Cheng
//====================================================================

// this one is special -- we break the grad into blocks
// grad is the actual filter in this conv
#define BLOCK_DIM_X   14
#define BLOCK_DIM_Y   14
#define GRAD_DIM      28
#define FILTER_DIM     5
#define NUM_FILTERS    6

#define IMAP_DIM_X (BLOCK_DIM_X + FILTER_DIM - 1)
#define IMAP_DIM_Y (BLOCK_DIM_Y + FILTER_DIM - 1)

#include <kernel_common.hpp>
#include <kernel_conv_baseline.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_conv_back_weight(
    hb_tensor_t* output,
    hb_tensor_t* input,
    hb_tensor_t* weight,
    hb_vector_t* padding,
    hb_vector_t* strides) {

    HBTensor<float> filter(output);
    HBTensor<float> imap(input);
    HBTensor<float> grad(weight);

    // extract parameters
    auto N      = filter.dim(0); // number of filters to calculate grad for
    auto Cout   = filter.dim(1); // number of channels in the images
    auto Hout   = filter.dim(2);
    auto Wout   = filter.dim(3);   // filter dimensions
    auto N_imap = imap.dim(0);   // number of images
    auto Hin    = imap.dim(2);   // image dimensions
    auto Win    = imap.dim(3);
    auto Hk     = grad.dim(2);   // grad dimensions
    auto Wk     = grad.dim(3);

    // cross check
    hb_assert(GRAD_DIM == Hk);
    hb_assert(GRAD_DIM == Wk);
    hb_assert(FILTER_DIM == Hout);
    hb_assert(FILTER_DIM == Wout);
    hb_assert(NUM_FILTERS == N);

    float filter_buf[FILTER_DIM * FILTER_DIM];      //   5x5 * 4 = 100B
    float imap_buf[IMAP_DIM_X * IMAP_DIM_Y];        // 18x18 * 4 = 1296B
    float grad_buf[BLOCK_DIM_X * BLOCK_DIM_Y];      // 14x14 * 4 = 784B

    // Here we break grad into blocks
    size_t h_blocks_per_out_channel = Hk / BLOCK_DIM_Y;
    size_t w_blocks_per_out_channel = Wk / BLOCK_DIM_X;
    if (Hk % BLOCK_DIM_Y != 0) {
       h_blocks_per_out_channel++;
    }
    if (Wk % BLOCK_DIM_X != 0) {
      w_blocks_per_out_channel++;
    }
    size_t blocks_per_out_channel = h_blocks_per_out_channel * w_blocks_per_out_channel;
    size_t num_blocks = N * Cout; // parallel over filter x channel


    auto imapDMA = [&](size_t image_id, size_t channel_id, size_t block_x, size_t block_y) {
      size_t imap_x = block_x * BLOCK_DIM_X;
      size_t imap_y = block_y * BLOCK_DIM_Y;
      float* imap_src_base = (float*)imap.data_ptr();
      uint32_t* imap_src_strides = imap.get_strides();
      imap_src_base += image_id * imap_src_strides[0] + channel_id * imap_src_strides[1];
      imap_src_base += imap_y * imap_src_strides[2] + imap_x * imap_src_strides[3];
      size_t y_step = imap_src_strides[2];
      fill_imap_buffer<IMAP_DIM_X, IMAP_DIM_Y>(imap_src_base, imap_buf, y_step);
    };

    auto gradDMA = [&](size_t image_id, size_t channel_id, size_t block_x, size_t block_y) {
      size_t grad_x = block_x * BLOCK_DIM_X;
      size_t grad_y = block_y * BLOCK_DIM_Y;
      float* grad_src_base = (float*)grad.data_ptr();
      uint32_t* grad_src_strides = grad.get_strides();
      grad_src_base += image_id * grad_src_strides[0] + channel_id * grad_src_strides[1];
      grad_src_base += grad_y * grad_src_strides[2] + grad_x * grad_src_strides[3];
      size_t y_step = grad_src_strides[2];
      fill_imap_buffer<BLOCK_DIM_X, BLOCK_DIM_Y>(grad_src_base, grad_buf, y_step);
    };

    auto filterDMA_wb = [&](size_t filter_id, size_t channel_id) {
      bsg_attr_remote float* filter_dst_base = (float*)filter.data_ptr();
      uint32_t* filter_dst_strides = filter.get_strides();
      filter_dst_base += filter_id * filter_dst_strides[0] + channel_id * filter_dst_strides[1];
      bsg_unroll(FILTER_DIM * FILTER_DIM)
      for (size_t i = 0; i < FILTER_DIM * FILTER_DIM; i++) {
        filter_dst_base[i] = filter_buf[i];
      }
    };


    bsg_cuda_print_stat_kernel_start();

    // main loop
    for (size_t idx = bsg_id; idx < num_blocks; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
      if (idx < num_blocks) {

        // figure out what we are producing
        size_t filter_id = idx / Cout;
        size_t channel_id = idx % Cout;

        // reset output buffer
        reset_buffer<FILTER_DIM, FILTER_DIM>(filter_buf);

        for (size_t image_id = 0; image_id < N_imap; image_id++) {
          for (size_t block_y = 0; block_y < h_blocks_per_out_channel; block_y++) {
            for (size_t block_x = 0; block_x < w_blocks_per_out_channel; block_x++) {

              // read in the image
              imapDMA(image_id, channel_id, block_x, block_y);

              // read in the grad
              gradDMA(image_id, filter_id, block_x, block_y);

              // do conv
              // 5x5 is too large to unroll in both x and y directions
              for (size_t f_y = 0; f_y < FILTER_DIM; f_y++) {
                register float psum0 = 0;
                register float psum1 = 0;
                register float psum2 = 0;
                register float psum3 = 0;
                register float psum4 = 0;
                float* imap_ptr = imap_buf + f_y * IMAP_DIM_X;
                float* grad_ptr = grad_buf;
                float* output = filter_buf + f_y * FILTER_DIM;
                for (size_t y = 0; y < BLOCK_DIM_Y; y++) {
                  float *imap_row = imap_ptr;
                  float *grad_row = grad_ptr;
                  bsg_unroll(4)
                  for (size_t x = 0; x < BLOCK_DIM_X; x++) {
                    psum0 += imap_row[x+0] * grad_row[x];
                    psum1 += imap_row[x+1] * grad_row[x];
                    psum2 += imap_row[x+2] * grad_row[x];
                    psum3 += imap_row[x+3] * grad_row[x];
                    psum4 += imap_row[x+4] * grad_row[x];
                  }
                  imap_ptr += IMAP_DIM_X;
                  grad_ptr += BLOCK_DIM_X;
                }
                output[0] += psum0;
                output[1] += psum1;
                output[2] += psum2;
                output[3] += psum3;
                output[4] += psum4;
              }

            } // block
          }
        } // image

        // write omap back
        filterDMA_wb(filter_id, channel_id);

      } // if (idx < num_blocks)
    } // main loop

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_conv_back_weight, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_vector_t*, hb_vector_t*)

}

