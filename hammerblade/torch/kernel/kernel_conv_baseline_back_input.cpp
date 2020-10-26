//====================================================================
// SPMD 2D Convolution
// Idea is that each tile will receive a piece of output image that
// does not overlap with any other tile to work on
// 10/02/2020 Lin Cheng
//====================================================================

#define BLOCK_DIM_X   16
#define BLOCK_DIM_Y   16
#define PADDING_X      4
#define PADDING_Y      4
#define FILTER_DIM     5
#define NUM_FILTERS    1

#define IMAP_DIM_X (BLOCK_DIM_X + FILTER_DIM - 1)
#define IMAP_DIM_Y (BLOCK_DIM_Y + FILTER_DIM - 1)

#include <kernel_common.hpp>
#include <kernel_conv_baseline.hpp>


extern "C" {

  __attribute__ ((noinline))  int tensorlib_conv_baseline_back_input(
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

    size_t h_blocks_per_out_channel = Hout / BLOCK_DIM_Y;
    size_t w_blocks_per_out_channel = Wout / BLOCK_DIM_X;
    if (Hout % BLOCK_DIM_Y != 0) {
      h_blocks_per_out_channel++;
    }
    if (Wout % BLOCK_DIM_X != 0) {
      w_blocks_per_out_channel++;
    }
    size_t blocks_per_out_channel = h_blocks_per_out_channel * w_blocks_per_out_channel;
    size_t num_blocks = N * Cout * blocks_per_out_channel;

    float filter_buf[FILTER_DIM * FILTER_DIM];  //   5x5 * 4 = 100B
    float omap_buf[BLOCK_DIM_X * BLOCK_DIM_Y];      // 14x14 * 4 = 784B
    float imap_buf[IMAP_DIM_X * IMAP_DIM_Y];        // 18x18 * 4 = 1296B

    // cross check
    hb_assert(FILTER_DIM == Hk);
    hb_assert(FILTER_DIM == Wk);
    hb_assert(NUM_FILTERS == Cout);


    // this one reads the filter in forward order
    // then write it into SPM is rotated order
    auto filterDMA_rotate = [&](size_t filter_id, size_t channel_id) {
      float* filter_src_base = (float*)filter.data_ptr();
      uint32_t* filter_src_strides = filter.get_strides();
      filter_src_base += filter_id * filter_src_strides[0] + channel_id * filter_src_strides[1];
      fill_filter_buffer_rotate<FILTER_DIM>(filter_src_base, filter_buf);
    };

    auto addPadding = [&](size_t start, size_t row, size_t col, size_t y_step) {
      for (size_t r = 0; r < row; r++) {
        size_t offset = start;
        for (size_t c = 0; c < col; c++) {
          imap_buf[offset] = 0;
          offset++;
        }
        start += y_step;
      }
    };

    auto imapDMA_padding = [&](size_t image_id, size_t channel_id, size_t block_x, size_t block_y) {
      size_t imap_x = block_x * BLOCK_DIM_X;
      size_t imap_y = block_y * BLOCK_DIM_Y;
      // this is used to correct the padding output offset
      imap_x = imap_x == 0 ? 0 : imap_x - PADDING_X;
      imap_y = imap_y == 0 ? 0 : imap_y - PADDING_Y;
      size_t logical_start = 0; // starting offset of imap buffer writting
      size_t read_x = 0;        // how many elements to read for a row
      size_t read_y = 0;        // how many rows to read
      // see if we need to add padding
      if (block_y == 0) {
        size_t imap_buf_offset = 0;
        // add top padding
        addPadding(0, PADDING_Y, IMAP_DIM_X, IMAP_DIM_X);
        if (block_x == 0) {
          // add left padding
          addPadding(PADDING_Y*IMAP_DIM_X, IMAP_DIM_Y-PADDING_Y, PADDING_X, IMAP_DIM_X);
          // setup read data copy
          logical_start = PADDING_Y*IMAP_DIM_X+PADDING_X;
          read_x = IMAP_DIM_X-PADDING_X;
          read_y = IMAP_DIM_Y-PADDING_Y;
        }
        else if (block_x == w_blocks_per_out_channel-1) {
          // add right padding
          addPadding(PADDING_Y*IMAP_DIM_X+IMAP_DIM_X-PADDING_X, IMAP_DIM_Y-PADDING_Y, PADDING_X, IMAP_DIM_X);
          logical_start = PADDING_Y*IMAP_DIM_X;
          read_x = IMAP_DIM_X-PADDING_X;
          read_y = IMAP_DIM_Y-PADDING_Y;
        } else {
          // top only
          logical_start = PADDING_Y*IMAP_DIM_X;
          read_x = IMAP_DIM_X;
          read_y = IMAP_DIM_Y-PADDING_Y;
        }
      } else if (block_y == h_blocks_per_out_channel-1) {
        // add bottom padding
        addPadding((IMAP_DIM_Y-PADDING_Y)*IMAP_DIM_X, PADDING_Y, IMAP_DIM_X, IMAP_DIM_X);
        if (block_x == 0) {
          // add left padding
          addPadding(0, IMAP_DIM_Y-PADDING_Y, PADDING_X, IMAP_DIM_X);
          logical_start = PADDING_X;
          read_x = IMAP_DIM_X-PADDING_X;
          read_y = IMAP_DIM_Y-PADDING_Y;
        }
        else if (block_x == w_blocks_per_out_channel-1) {
          // add right padding
          addPadding(IMAP_DIM_X-PADDING_X, IMAP_DIM_Y-PADDING_Y, PADDING_X, IMAP_DIM_X);
          logical_start = 0;
          read_x = IMAP_DIM_X-PADDING_X;
          read_y = IMAP_DIM_Y-PADDING_Y;
        } else {
          // bottom only
          logical_start = 0;
          read_x = IMAP_DIM_X;
          read_y = IMAP_DIM_Y-PADDING_Y;
        }
      } else if (block_x == 0) {
        // add left padding only
        addPadding(0, IMAP_DIM_Y, PADDING_X, IMAP_DIM_X);
        logical_start = PADDING_X;
        read_x = IMAP_DIM_X-PADDING_X;
        read_y = IMAP_DIM_Y;
      } else if (block_x == w_blocks_per_out_channel-1) {
        // add right padding only
        addPadding(IMAP_DIM_X-PADDING_X, IMAP_DIM_Y, PADDING_X, IMAP_DIM_X);
        logical_start = 0;
        read_x = IMAP_DIM_X-PADDING_X;
        read_y = IMAP_DIM_Y;
      } else {
        // no padding is needed
        logical_start = 0;
        read_x = IMAP_DIM_X;
        read_y = IMAP_DIM_Y;
      }
      float* imap_src_base = (float*)imap.data_ptr();
      uint32_t* imap_src_strides = imap.get_strides();
      imap_src_base += image_id * imap_src_strides[0] + channel_id * imap_src_strides[1];
      imap_src_base += imap_y * imap_src_strides[2] + imap_x * imap_src_strides[3];
      size_t y_step = imap_src_strides[2];
      //fill_imap_buffer<IMAP_DIM_X, IMAP_DIM_Y>(imap_src_base, imap_buf, y_step);
      for (size_t r = 0; r < read_y; r++) {
        size_t row_offset = logical_start;
        float* row_src = imap_src_base;
        for (size_t c = 0; c < read_x; c++) {
          imap_buf[row_offset] = *row_src;
          row_src++;
          row_offset++;
        }
        imap_src_base += y_step;
        logical_start += IMAP_DIM_X;
      }
      // debug
      size_t debug_offset = 0;
      for (size_t r = 0; r < IMAP_DIM_Y; r++) {
        for (size_t c = 0; c < IMAP_DIM_X; c++) {
          std::cout << imap_buf[debug_offset] << " ";
          debug_offset++;
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
      std::cout << std::endl;
    };

    auto omapDMA = [&](size_t image_id, size_t filter_id, size_t block_x, size_t block_y) {
      size_t omap_x = block_x * BLOCK_DIM_X;
      size_t omap_y = block_y * BLOCK_DIM_Y;
      float* omap_src_base = (float*)omap.data_ptr();
      uint32_t* omap_src_strides = omap.get_strides();
      omap_src_base += image_id * omap_src_strides[0] + filter_id * omap_src_strides[1];
      omap_src_base += omap_y * omap_src_strides[2] + omap_x * omap_src_strides[3];
      size_t y_step = omap_src_strides[2];
      drain_omap_buffer<BLOCK_DIM_X, BLOCK_DIM_Y>(omap_buf, omap_src_base, y_step);
    };

    bsg_cuda_print_stat_kernel_start();

    // main loop
    for (size_t idx = bsg_id; idx < num_blocks; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
      if (idx < num_blocks) {

        // figure out what we are producing
        size_t tmp = idx;
        size_t image_id = tmp / (Cout * blocks_per_out_channel);
        tmp = tmp % (Cout * blocks_per_out_channel);
        size_t filter_id = tmp / blocks_per_out_channel;
        tmp = tmp % blocks_per_out_channel;
        size_t block_y = tmp / w_blocks_per_out_channel;
        size_t block_x = tmp % w_blocks_per_out_channel;

        // reset output buffer
        reset_buffer<BLOCK_DIM_X, BLOCK_DIM_Y>(omap_buf);

        for (size_t channel_id = 0; channel_id < Cin; channel_id++) {

          // read in the image
          imapDMA_padding(image_id, channel_id, block_x, block_y);

          // read in the filter
          filterDMA_rotate(filter_id, channel_id);

          // do conv
          conv2d_5x5_back(imap_buf, filter_buf, omap_buf);

        } // channel

        // write omap back
        omapDMA(image_id, filter_id, block_x, block_y);

      } // if (idx < num_blocks)
    } // main loop

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_conv_baseline_back_input, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_vector_t*, hb_vector_t*)

}

