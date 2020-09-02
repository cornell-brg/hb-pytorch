//====================================================================
// Experimental block level systolic array for GEMM
// 08/13/2020 Lin Cheng
//====================================================================

#include <kernel_common.hpp>

// Eyeriss buffer setup
#define FILTER_BUF_SIZE 128
#define   IMAP_BUF_SIZE 256
#define   PSUM_BUF_SIZE  64

extern "C" {

  __attribute__ ((noinline))  int tensorlib_eyeriss(
    hb_tensor_t* output,
    hb_tensor_t* input,
    hb_tensor_t* weight,
    hb_vector_t* padding,
    hb_vector_t* strides) {


    HBTensor<float> omap(output);
    HBTensor<float> imap(input);
    HBTensor<float> filter(weight);

    //   imap[#images][#in__channel][row][col]
    //   omap[#images][#out_channel][row][col]
    // filter[#filter][#in__channel][ROW][COL]

    // Conv2d parameters
    auto N    = omap.dim(0); // number of minibatches
    auto Cout = omap.dim(1); // number of output channels
    auto Hout = omap.dim(2);
    auto Wout = omap.dim(3);
    auto Cin  = imap.dim(1); // number of input channels
    auto Hin  = imap.dim(2);
    auto Win  = imap.dim(3);
    auto Kh   = filter.dim(2);
    auto Kw   = filter.dim(3);

    // Eyeriss buffers
    float filter_buf[FILTER_BUF_SIZE];
    float   imap_buf[IMAP_BUF_SIZE];
    float   psum_buf[PSUM_BUF_SIZE];

    // config
    // 0 -- idle       -- do nothing
    // 1 -- filter DMA -- push to 2 to the east
    // 2 -- imap DMA   -- push to NE
    // 3 -- compute    -- push to NE

    char 2x2_eyeriss_config[4][4] = {
        {1, 0, 3, 3},
        {1, 2, 3, 3},
        {0, 2, 2, 0},
        {0, 0, 0, 0}
    };

    // active config
    char** mc_config = 2x2_eyeriss_config;

    bsg_cuda_print_stat_kernel_start();

    // tile task dispatch
    char tile_config = mc_config[bsg_x][bsg_y];
    switch (tile_config) {
      case 0:
        // nothing
        break;
      case 1:
        // filter DMA
        break;
      case 2:
        // imap DMA
        break;
      case 3:
        // compute
        break;
      default:
        hb_assert_msg(false, "invalid tile task config");
    }

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_eyeriss, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}

