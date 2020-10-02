//====================================================================
// SPMD 2D Convolution
// Idea is that each tile will receive a piece of output image that
// does not overlap with any other tile to work on
// 10/02/2020 Lin Cheng
//====================================================================

#include <kernel_common.hpp>

#define BLOCK_DIM   14
#define FILTER_DIM   5
#define NUM_FILTERS  6

#define IMAP_DIM (BLOCK_DIM + FILTER_DIM - 1)

extern "C" {

  __attribute__ ((noinline))  int tensorlib_conv_baseline(
    hb_tensor_t* output,
    hb_tensor_t* input,
    hb_tensor_t* weight,
    hb_vector_t* padding,
    hb_vector_t* strides) {

    HBTensor<float> omap(output);
    HBTensor<float> imap(input);
    HBTensor<float> filter(weight);

    // Conv2d parameters
    auto N    = omap.dim(0); // number of images in batch
    auto Cout = omap.dim(1); // number of output channels
    auto Hout = omap.dim(2);
    auto Wout = omap.dim(3);
    auto Cin  = imap.dim(1); // number of input channels
    auto Hin  = imap.dim(2);
    auto Win  = imap.dim(3);
    auto Hk   = filter.dim(2);
    auto Wk   = filter.dim(3);

    size_t h_blocks_per_out_channel = Hout / BLOCK_DIM;
    size_t w_blocks_per_out_channel = Wout / BLOCK_DIM;
    if (Hout % BLOCK_DIM != 0) {
      h_blocks_per_out_channel++;
    }
    if (Wout % BLOCK_DIM != 0) {
      w_blocks_per_out_channel++;
    }
    size_t blocks_per_out_channel = h_blocks_per_out_channel * w_blocks_per_out_channel;
    size_t num_blocks = N * Cout * blocks_per_out_channel;

    float filter_buf[FILTER_DIM * FILTER_DIM];  //   5x5 * 4 = 100B
    float omap_buf[BLOCK_DIM * BLOCK_DIM];      // 14x14 * 4 = 784B
    float imap_buf[IMAP_DIM * IMAP_DIM];        // 18x18 * 4 = 1296B

    // cross check
    hb_assert(FILTER_DIM == Hk);
    hb_assert(FILTER_DIM == Wk);
    hb_assert(NUM_FILTERS == Cout);


    bsg_cuda_print_stat_kernel_start();


    std::cout << "(BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM) = " << (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM) << std::endl;

    // main loop
    for (size_t idx = bsg_id; idx < num_blocks; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
      if (idx < num_blocks) {

        // figure out what we are producing
        size_t tmp = idx;
        size_t image_id = tmp / (Cout * blocks_per_out_channel);
        tmp = tmp % (Cout * blocks_per_out_channel);
        size_t filter_id = tmp / blocks_per_out_channel;
        tmp = tmp % blocks_per_out_channel;
        size_t block_y = tmp / h_blocks_per_out_channel;
        size_t block_x = tmp % h_blocks_per_out_channel;

        // reset output buffer
        for (size_t buf_idx = 0; buf_idx < BLOCK_DIM * BLOCK_DIM; buf_idx++) {
          omap_buf[buf_idx] = 0;
        }

        for (size_t channel_id = 0; channel_id < Cin; channel_id++) {
          // read in the image
          size_t imap_x = block_x * BLOCK_DIM;
          size_t imap_x_end = imap_x + IMAP_DIM;
          size_t imap_y = block_y * BLOCK_DIM;
          size_t imap_y_end = imap_y + IMAP_DIM;

          size_t buf_idx = 0;
          for (;imap_y < imap_y_end; imap_y++) {
            for (size_t imap_xi = imap_x; imap_xi < imap_x_end; imap_xi++) {
              imap_buf[buf_idx] = imap(image_id, channel_id, imap_y, imap_xi);
              buf_idx++;
            }
          }

          // read in the filter
          buf_idx = 0;
          for (size_t filter_y = 0; filter_y < FILTER_DIM; filter_y++) {
            for (size_t filter_x = 0; filter_x < FILTER_DIM; filter_x++) {
              filter_buf[buf_idx] = filter(filter_id, channel_id, filter_y, filter_x);
              buf_idx++;
            }
          }

          // do naive conv 2D on these buffers
          for (size_t y = 0; y < BLOCK_DIM; y++) {
            for (size_t x = 0; x < BLOCK_DIM; x++) {
              float psum = 0;
              for (size_t yy = 0; yy < FILTER_DIM; yy++) {
                for (size_t xx = 0; xx < FILTER_DIM; xx++) {
                  psum += filter_buf[yy * FILTER_DIM + xx] * imap_buf[y * IMAP_DIM + x + yy * IMAP_DIM + xx];
                }
              }
              omap_buf[y * BLOCK_DIM + x] += psum;
            }
          }

        } // channel

        // write omap back
        size_t omap_x = block_x * BLOCK_DIM;
        size_t omap_x_end = omap_x + BLOCK_DIM;
        size_t omap_y = block_y * BLOCK_DIM;
        size_t omap_y_end = omap_y + BLOCK_DIM;

        size_t buf_idx = 0;
        for (;omap_y < omap_y_end; omap_y++) {
          for (size_t omap_xi = omap_x; omap_xi < omap_x_end; omap_xi++) {
            omap(image_id, filter_id, omap_y, omap_xi) = omap_buf[buf_idx];
            buf_idx++;
          }
        }
      } // if (idx < num_blocks)
    } // main loop

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_conv_baseline, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_vector_t*, hb_vector_t*)

}

