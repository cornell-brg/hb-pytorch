//====================================================================
// Experimental block level systolic array for GEMM
// 08/13/2020 Lin Cheng
//====================================================================

#include <kernel_common.hpp>

// Eyeriss buffer setup
#define FILTER_BUF_SIZE 128
#define   IMAP_BUF_SIZE 256
#define   PSUM_BUF_SIZE  64

// Eyeriss config
// we use filter-use scheme -- filter stays constant within a process pass
#define IMAGES_PER_BURST 1
#define FILTERS_PER_PROCESSING_PASS 2

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

    // Eyeriss buffers
    //
    //   imap[#images]     [#in_channel] [row][col]
    //   omap[#images]     [#out_channel][row][col]
    // filter[#out_channel][#in_channel] [ROW][COL]

    float filter_buf[FILTER_BUF_SIZE];
    float   imap_buf[IMAP_BUF_SIZE];
    float   psum_buf[PSUM_BUF_SIZE];

    // sync flags
    // 0 -> ready to load
    // 1 -> ready to use

    /*
    volatile unsigned int  filter_A_f      = 0;
    volatile unsigned int  filter_A_f_E    = 0;
    volatile unsigned int *filter_A_f_E_r  = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y,&filter_A_f));
    volatile unsigned int *filter_A_f_W_r  = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x-1,bsg_y,&filter_A_f_E));

    volatile unsigned int  psum_A_f        = 0;
    volatile unsigned int  psum_A_f_N      = 0;
    volatile unsigned int *psum_A_f_N_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y-1,&psum_A_f));
    volatile unsigned int *psum_A_f_S_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y+1,&psum_A_f_N));

    volatile unsigned int  imap_A_f        = 0;
    volatile unsigned int  imap_A_f_NE     = 0;
    volatile unsigned int *imap_A_f_NE_r   = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y-1,&imap_A_f));
    volatile unsigned int *imap_A_f_SW_r   = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x-1,bsg_y+1,&imap_A_f_NE));

    // proxy flags for supporting double buffering

    volatile unsigned int *filter_f        = &filter_A_f;
    volatile unsigned int *filter_f_E      = &filter_A_f_E;
    volatile unsigned int *filter_f_E_r    = filter_A_f_E_r;
    volatile unsigned int *filter_f_W_r    = filter_A_f_W_r;

    volatile unsigned int *psum_f          = &psum_A_f;
    volatile unsigned int *psum_f_N        = & psum_A_f_N;
    volatile unsigned int *psum_f_N_r      = psum_A_f_N_r;
    volatile unsigned int *psum_f_S_r      = psum_A_f_S_r;

    volatile unsigned int *imap_f          = &imap_A_f;
    volatile unsigned int *imap_f_NE       = imap_A_f_NE;
    volatile unsigned int *imap_f_NE_r     = imap_A_f_NE_r;
    volatile unsigned int *imap_f_SW_r     = imap_A_f_SW_r;
    */

    // Conv2d parameters
    auto N    = omap.dim(0); // number of minibatches
    auto Cout = omap.dim(1); // number of output channels
    auto Hout = omap.dim(2);
    auto Wout = omap.dim(3);
    auto Cin  = imap.dim(1); // number of input channels
    auto Hin  = imap.dim(2);
    auto Win  = imap.dim(3);
    auto Hk   = filter.dim(2);
    auto Wk   = filter.dim(3);

    // std::cout << "N = "    << N << std::endl;
    // std::cout << "Cout = " << Cout << std::endl;
    // std::cout << "Hout = " << Hout << std::endl;
    // std::cout << "Wout = " << Wout << std::endl;
    // std::cout << "Cin = "  << Cin << std::endl;
    // std::cout << "Hin = "  << Hin << std::endl;
    // std::cout << "Win = "  << Win << std::endl;
    // std::cout << "Hk = "   << Hk << std::endl;
    // std::cout << "Wk = "   << Wk << std::endl;

    // config
    // 0 -- idle       -- do nothing
    // 1 -- filter DMA -- push to 2 to the east
    // 2 -- imap DMA   -- push to NE
    // 3 -- compute    -- push to NE

    char eyeriss_2x2_config[4][4] = {
        {1, 0, 3, 3},
        {1, 2, 3, 3},
        {0, 2, 2, 0},
        {0, 0, 0, 0}
    };

    char debug_config[4][4] = {
        {0, 0, 3, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    };

    // active config
    char (&mc_config)[4][4] = debug_config;

    bsg_cuda_print_stat_kernel_start();

    // tile task dispatch
    char tile_config = mc_config[bsg_x][bsg_y];
    switch (tile_config) {
      case 0:
        // nothing
        break;
      case 1:
        // filter DMA
        for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {
          size_t buf_offset = 0;
          for (size_t filter_id = 0; filter_id < FILTERS_PER_PROCESSING_PASS; filter_id++) {
            // TODO -- channel
            for (size_t col = 0; col < Wk; col++) {
              filter_buf[buf_offset] = filter(filter_id+filters,0,bsg_y,col);
              std::cout << "last filter data copied - " << filter_buf[buf_offset] << std::endl;
              buf_offset++;
            }
          }
          std::cout << " -- end of a pass -- " << std::endl;
        }
        break;
      case 2:
        // imap DMA
        for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {
          for (size_t images = 0; images < N; images += IMAGES_PER_BURST) {
            size_t buf_offset = 0;
            for (size_t image_id = 0; image_id < IMAGES_PER_BURST; image_id++) {
              // TODO -- channel
              for (size_t col = 0; col < Win; col++) {
                imap_buf[buf_offset] = imap(image_id+images,0,(bsg_x-1)+(bsg_y-1),col);
                std::cout << "last imap data copied - " << imap_buf[buf_offset] << std::endl;
                buf_offset++;
              }
            }
          }
          std::cout << " -- end of a pass -- " << std::endl;
        }
        break;
      case 3:
        // compute
        for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {
          for (size_t images = 0; images < N; images += IMAGES_PER_BURST) {
            for (size_t image_id = 0; image_id < IMAGES_PER_BURST; image_id++) {
              for (size_t filter_id = 0; filter_id < FILTERS_PER_PROCESSING_PASS; filter_id++) {
                // compute for each (filter,image) pair
                std::cout << "compute (" << filter_id + filters << ", " << images + image_id << ")" << std::endl;
              }
            }
          }
          std::cout << " -- end of a pass -- " << std::endl;
        }
        break;
      default:
        hb_assert_msg(false, "invalid tile task config");
    }

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_eyeriss, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_vector_t*, hb_vector_t*)

}

