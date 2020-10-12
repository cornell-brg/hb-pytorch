//====================================================================
// Output stationary systolic style Conv2D
// Handcrafted for LeNet-5 Conv1
// which has only 1 channel
// It shares the same compute core with conv_baseline
// 10/12/2020 Lin Cheng
//====================================================================

// Layer setup
#define BLOCK_DIM   14
#define FILTER_DIM   5
#define NUM_FILTERS  6
#define IMAP_DIM (BLOCK_DIM + FILTER_DIM - 1)

#include <kernel_common.hpp>
#include <kernel_conv_baseline.hpp>

extern "C" {
  __attribute__ ((noinline))  int tensorlib_conv_systolic(
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

    // Buffers
    float filter_buf[FILTER_DIM * FILTER_DIM];  //   5x5 * 4 = 100B
    float omap_buf[BLOCK_DIM * BLOCK_DIM];      // 14x14 * 4 = 784B
    float imap_buf[IMAP_DIM * IMAP_DIM];        // 18x18 * 4 = 1296B

    // Config
    // 0 -- idle
    // 1 -- imap DMA
    // 2 -- compute
    // 3 -- polyA stoppper

    // Spatially compute 4 images at once
    // image 0
    // DMA --  upper-left -- stopper  DMA --  upper-right -- stopper
    // DMA -- bottom-left -- stopper  DMA -- bottom-right -- stopper
    // image 1
    // DMA --  upper-left -- stopper  DMA --  upper-right -- stopper
    // DMA -- bottom-left -- stopper  DMA -- bottom-right -- stopper
    // image 2
    // DMA --  upper-left -- stopper  DMA --  upper-right -- stopper
    // DMA -- bottom-left -- stopper  DMA -- bottom-right -- stopper
    // image 3
    // DMA --  upper-left -- stopper  DMA --  upper-right -- stopper
    // DMA -- bottom-left -- stopper  DMA -- bottom-right -- stopper
    char systolic_lenet[8][16] = {
      {1, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 2, 3},
      {1, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 2, 3},
      {1, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 2, 3},
      {1, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 2, 3},
      {1, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 2, 3},
      {1, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 2, 3},
      {1, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 2, 3},
      {1, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 2, 2, 3},
    };

    char tile_config = systolic_lenet[bsg_y][bsg_x];

    // Helper functions

    auto filterDMA = [&](size_t filter_id, size_t channel_id) {
      float* filter_src_base = (float*)filter.data_ptr();
      uint32_t* filter_src_strides = filter.get_strides();
      filter_src_base += filter_id * filter_src_strides[0] + channel_id * filter_src_strides[1];
      fill_filter_buffer<FILTER_DIM>(filter_src_base, filter_buf);
    };

    auto imapDMA = [&](size_t image_id, size_t channel_id, size_t block_x, size_t block_y) {
      size_t imap_x = block_x * BLOCK_DIM;
      size_t imap_y = block_y * BLOCK_DIM;
      float* imap_src_base = (float*)imap.data_ptr();
      uint32_t* imap_src_strides = imap.get_strides();
      imap_src_base += image_id * imap_src_strides[0] + channel_id * imap_src_strides[1];
      imap_src_base += imap_y * imap_src_strides[2] + imap_x * imap_src_strides[3];
      size_t y_step = imap_src_strides[2];
      fill_imap_buffer<IMAP_DIM>(imap_src_base, imap_buf, y_step);
    };

    auto omapDMA = [&](size_t image_id, size_t filter_id, size_t block_x, size_t block_y) {
      size_t omap_x = block_x * BLOCK_DIM;
      size_t omap_y = block_y * BLOCK_DIM;
      float* omap_src_base = (float*)omap.data_ptr();
      uint32_t* omap_src_strides = omap.get_strides();
      omap_src_base += image_id * omap_src_strides[0] + filter_id * omap_src_strides[1];
      omap_src_base += omap_y * omap_src_strides[2] + omap_x * omap_src_strides[3];
      size_t y_step = omap_src_strides[2];
      drain_omap_buffer<BLOCK_DIM>(omap_buf, omap_src_base, y_step);
    };

    auto compute_job = [&]() {
      // XXX: this works for single input channel only
      size_t filter_id = bsg_x % 8 - 1;
      size_t block_x = bsg_x / 8;
      size_t block_y = bsg_y % 2;
      size_t channel_id = 0;
      size_t image_offset = bsg_y / 2;

      filterDMA(filter_id, channel_id);

      for (size_t idx = 0; idx < N; idx += 4) {
        size_t image_id = image_offset + idx;
        // wait for imap
        // pass imap

        // do compute
        conv2d_5x5(imap_buf, filter_buf, omap_buf);

        // write omap back
        omapDMA(image_id, filter_id, block_x, block_y);
      }
    };

    auto imapDMA_job = [&]() {
      imapDMA(image_id, channel_id, block_x, block_y);
      // pass imap
    };
    auto polyA_job = [&]() {
      // wait for imap
    };


    // Job dispatch

    bsg_cuda_print_stat_kernel_start();

    switch (tile_config) {
      case 0:
        // nothing
        break;
      case 1:
        imapDMA_job();
        break;
      case 2:
        compute_job();
        break;
      case 3:
        polyA_job();
        break;
      default:
        hb_assert_msg(false, "invalid tile task config");
    }

    bsg_cuda_print_stat_kernel_end();
    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_conv_systolic, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_vector_t*, hb_vector_t*)

}
