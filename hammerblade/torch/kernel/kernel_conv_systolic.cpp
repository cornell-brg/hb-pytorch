//====================================================================
// Output stationary systolic style Conv2D
// Handcrafted for LeNet-5 Conv1
// which has only 1 channel
// It shares the same compute core with conv_baseline
// 10/12/2020 Lin Cheng
//====================================================================

// Layer setup
#define BLOCK_DIM_X   14
#define BLOCK_DIM_Y   14
#define FILTER_DIM     5
#define NUM_FILTERS    6
#define IMAP_DIM_X    (BLOCK_DIM_X + FILTER_DIM - 1)
#define IMAP_DIM_Y    (BLOCK_DIM_Y + FILTER_DIM - 1)

#include <kernel_common.hpp>
#include <kernel_conv_baseline.hpp>
#include <kernel_circular_buffer.hpp>

inline void spcpy(float* dest, float* src) {
  for (int i = 0; i < IMAP_DIM_X * IMAP_DIM_Y; i += 9) {
        register float tmp0 = *(src + 0);
        register float tmp1 = *(src + 1);
        register float tmp2 = *(src + 2);
        register float tmp3 = *(src + 3);
        register float tmp4 = *(src + 4);
        register float tmp5 = *(src + 5);
        register float tmp6 = *(src + 6);
        register float tmp7 = *(src + 7);
        register float tmp8 = *(src + 8);
        asm volatile("": : :"memory");
        *(dest + 0) = tmp0;
        *(dest + 1) = tmp1;
        *(dest + 2) = tmp2;
        *(dest + 3) = tmp3;
        *(dest + 4) = tmp4;
        *(dest + 5) = tmp5;
        *(dest + 6) = tmp6;
        *(dest + 7) = tmp7;
        *(dest + 8) = tmp8;
        src += 9;
        dest += 9;
  }
}

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
    float  filter_buf[FILTER_DIM * FILTER_DIM];  //   5x5 * 4 = 100B
    float  omap_buf[BLOCK_DIM_X * BLOCK_DIM_Y];  // 14x14 * 4 = 784B
    float* imap_buf;

    // Buffer
    CircularBuffer::FIFO<float, IMAP_DIM_X * IMAP_DIM_Y, 1> fifo(bsg_y, bsg_x-1, bsg_y, bsg_x+1);

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

    auto imapDMA = [&](float* buf, size_t image_id, size_t channel_id, size_t block_x, size_t block_y) {
      size_t imap_x = block_x * BLOCK_DIM_X;
      size_t imap_y = block_y * BLOCK_DIM_Y;
      float* imap_src_base = (float*)imap.data_ptr();
      uint32_t* imap_src_strides = imap.get_strides();
      imap_src_base += image_id * imap_src_strides[0] + channel_id * imap_src_strides[1];
      imap_src_base += imap_y * imap_src_strides[2] + imap_x * imap_src_strides[3];
      size_t y_step = imap_src_strides[2];
      // fill_imap_buffer<IMAP_DIM_X, IMAP_DIM_Y>(imap_src_base, buf, y_step);
      // bsg_attr_remote is not doing its job here ... idk why
      // it works for baseline
      size_t buf_offset = 0;
      for (size_t i = 0; i < IMAP_DIM_Y; i++) {
        register float tmp00 = *(imap_src_base + 0);
        register float tmp01 = *(imap_src_base + 1);
        register float tmp02 = *(imap_src_base + 2);
        register float tmp03 = *(imap_src_base + 3);
        register float tmp04 = *(imap_src_base + 4);
        register float tmp05 = *(imap_src_base + 5);
        register float tmp06 = *(imap_src_base + 6);
        register float tmp07 = *(imap_src_base + 7);
        register float tmp08 = *(imap_src_base + 8);
        register float tmp09 = *(imap_src_base + 9);
        register float tmp10 = *(imap_src_base + 10);
        register float tmp11 = *(imap_src_base + 11);
        register float tmp12 = *(imap_src_base + 12);
        register float tmp13 = *(imap_src_base + 13);
        register float tmp14 = *(imap_src_base + 14);
        register float tmp15 = *(imap_src_base + 15);
        register float tmp16 = *(imap_src_base + 16);
        register float tmp17 = *(imap_src_base + 17);
        asm volatile("": : :"memory");
        buf[buf_offset + 0]  = tmp00;
        buf[buf_offset + 1]  = tmp01;
        buf[buf_offset + 2]  = tmp02;
        buf[buf_offset + 3]  = tmp03;
        buf[buf_offset + 4]  = tmp04;
        buf[buf_offset + 5]  = tmp05;
        buf[buf_offset + 6]  = tmp06;
        buf[buf_offset + 7]  = tmp07;
        buf[buf_offset + 8]  = tmp08;
        buf[buf_offset + 9]  = tmp09;
        buf[buf_offset + 10] = tmp10;
        buf[buf_offset + 11] = tmp11;
        buf[buf_offset + 12] = tmp12;
        buf[buf_offset + 13] = tmp13;
        buf[buf_offset + 14] = tmp14;
        buf[buf_offset + 15] = tmp15;
        buf[buf_offset + 16] = tmp16;
        buf[buf_offset + 17] = tmp17;

        buf_offset += IMAP_DIM_X;
        imap_src_base += y_step;
      }
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

        // reset output buffer
        reset_buffer<BLOCK_DIM_X, BLOCK_DIM_Y>(omap_buf);
        // wait for imap
        imap_buf = fifo.obtain_rd_ptr();
        // pass imap
        float* imap_buf_remote = fifo.obtain_wr_ptr();
        // copy
        spcpy(imap_buf_remote, imap_buf);

        fifo.finish_wr_ptr();

        // do compute
        conv2d_5x5(imap_buf, filter_buf, omap_buf);

        fifo.finish_rd_ptr();

        // write omap back
        omapDMA(image_id, filter_id, block_x, block_y);
      }
    };

    auto imapDMA_job = [&]() {
      imap_buf = fifo.get_buffer(); // reuse
      size_t block_x = bsg_x / 8;
      size_t block_y = bsg_y % 2;
      size_t channel_id = 0;
      size_t image_offset = bsg_y / 2;

      for (size_t idx = 0; idx < N; idx += 4) {
        size_t image_id = image_offset + idx;
        imapDMA(imap_buf, image_id, channel_id, block_x, block_y);
        // pass imap
        float* imap_buf_remote = fifo.obtain_wr_ptr();
        // copy
        spcpy(imap_buf_remote, imap_buf);
        fifo.finish_wr_ptr();
      }
    };

    auto polyA_job = [&]() {
      for (size_t idx = 0; idx < N; idx += 4) {
        // wait for imap
        fifo.obtain_rd_ptr();
        fifo.finish_rd_ptr();
      }
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
