//====================================================================
// Experimental block level systolic array for GEMM
// 08/13/2020 Lin Cheng
//====================================================================

#include <kernel_common.hpp>

// Eyeriss buffer setup
#define FILTER_BUF_SIZE 128
#define   IMAP_BUF_SIZE 256
#define   PSUM_BUF_SIZE  64
#define       LOAD_PSUM   0

// Eyeriss config
// we use filter-use scheme -- filter stays constant within a process pass
#define IMAGES_PER_BURST 1
#define FILTERS_PER_PROCESSING_PASS 2
#define EYERISS_ROW 2
#define EYERISS_COL 2

#define DEVICE_X (EYERISS_ROW + 2)
#define DEVICE_Y (EYERISS_COL + 2)
#define PASS_PSUM (bsg_y > 0)
#define PASS_IMAP (bsg_x < (DEVICE_X - 1) && bsg_y >0)
#define PASS_FILTER (bsg_x < (DEVICE_X - 1))

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

    float filter_buf_A[FILTER_BUF_SIZE];
    float   imap_buf_A[IMAP_BUF_SIZE];
    float   psum_buf_A[PSUM_BUF_SIZE];

    float *filter_buf = filter_buf_A;
    float   *imap_buf =   imap_buf_A;
    float   *psum_buf =   psum_buf_A;

    float *filter_buf_A_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y,filter_buf_A)); // East
    float   *imap_buf_A_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y-1,imap_buf_A)); // NorthEast
    float   *psum_buf_A_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y-1,psum_buf_A));   // North

    // filter DMA
    if (bsg_x == 0) {
      filter_buf_A_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x+2,bsg_y,filter_buf_A)); // East x 2
    }

    // psum DMA
    if (bsg_y == 3) {
      psum_buf_A_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y-2,psum_buf_A));   // North x 2
    }

    float *filter_buf_remote = filter_buf_A_remote;
    float   *imap_buf_remote =   imap_buf_A_remote;
    float   *psum_buf_remote =   psum_buf_A_remote;

    // sync flags
    // 0 -> ready to load
    // 1 -> ready to use

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

    // filter DMA
    if (bsg_x == 0) {
      filter_A_f_E_r = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x+2,bsg_y,&filter_A_f)); // East x 2
      filter_A_f_W_r = NULL;
    }

    // psum DMA
    if (bsg_y == 3) {
      psum_A_f_N_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y-2,&psum_A_f));  // North x 2
      psum_A_f_S_r = NULL;
    }

    // first col of PE
    if (bsg_x == 2) {
      filter_A_f_W_r  = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x-2,bsg_y,&filter_A_f_E)); // West x 2
    }

    // bottom row of PE
    if (bsg_y == 1) {
      psum_A_f_S_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y+2,&psum_A_f_N));  // South x 2
    }

    // proxy flags for supporting double buffering

    volatile unsigned int *filter_f        = &filter_A_f;
    volatile unsigned int *filter_f_E      = &filter_A_f_E;
    volatile unsigned int *filter_f_E_r    = filter_A_f_E_r;
    volatile unsigned int *filter_f_W_r    = filter_A_f_W_r;

    volatile unsigned int *psum_f          = &psum_A_f;
    volatile unsigned int *psum_f_N        = &psum_A_f_N;
    volatile unsigned int *psum_f_N_r      = psum_A_f_N_r;
    volatile unsigned int *psum_f_S_r      = psum_A_f_S_r;

    volatile unsigned int *imap_f          = &imap_A_f;
    volatile unsigned int *imap_f_NE       = &imap_A_f_NE;
    volatile unsigned int *imap_f_NE_r     = imap_A_f_NE_r;
    volatile unsigned int *imap_f_SW_r     = imap_A_f_SW_r;

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
    // 1 -- filter DMA -- push to 2 to the East
    // 2 -- imap DMA   -- push to NE
    // 3 -- psum DMA   -- push to 2 to the North
    // 4 -- compute    -- push to NE & N

    // char eyeriss_2x2_config[4][4] = {
    //     {1, 0, 4, 4},
    //     {1, 2, 4, 4},
    //     {0, 2, 2, 0},
    //     {0, 0, 3, 3}
    // };

    // char debug_config[4][4] = {
    //     {1, 0, 4, 4},
    //     {1, 2, 4, 4},
    //     {0, 2, 2, 0},
    //     {0, 0, 3, 3}
    // };

    char eyeriss_2x2_debug[8][16] = {
        {1, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    // active config
    char (&mc_config)[8][16] = eyeriss_2x2_debug;

    bsg_cuda_print_stat_kernel_start();

    // tile task dispatch
    char tile_config = mc_config[bsg_y][bsg_x];
    switch (tile_config) {
      case 0:
        // nothing
        break;
      case 1:
        // filter DMA
        for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {
          size_t buf_offset = 0;

          //bsg_printf("in filter DMA\n");
          // wait until remote filter buffer is ready
          bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (filter_f_E)), 0);
          //bsg_printf(" -- buffer ready\n");
          for (size_t filter_id = 0; filter_id < FILTERS_PER_PROCESSING_PASS; filter_id++) {
            // TODO -- channel
            for (size_t col = 0; col < Wk; col++) {
              filter_buf_remote[buf_offset] = filter(filter_id+filters,0,bsg_y,col);
              buf_offset++;
            }
          }
          asm volatile("": : :"memory");
          *filter_f_E = 1;
          *filter_f_E_r = 1;
          //bsg_printf(" -- buffer copying done\n");

          // std::cout << " -- end of a pass -- " << std::endl;
        }
        break;
      case 2:
        // imap DMA
        for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {
          for (size_t images = 0; images < N; images += IMAGES_PER_BURST) {
            size_t buf_offset = 0;

            //bsg_printf("in imap DMA\n");
            // wait until remote imap buffer is ready
            bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (imap_f_NE)), 0);
            //bsg_printf(" -- buffer ready\n");
            for (size_t image_id = 0; image_id < IMAGES_PER_BURST; image_id++) {
              // TODO -- channel
              for (size_t col = 0; col < Win; col++) {
                imap_buf_remote[buf_offset] = imap(image_id+images,0,(bsg_x-1)+(bsg_y-1),col);
                buf_offset++;
              }
            }
            asm volatile("": : :"memory");
            *imap_f_NE = 1;
            *imap_f_NE_r = 1;
            //bsg_printf(" -- buffer copying done\n");
          }
          // std::cout << " -- end of a pass -- " << std::endl;
        }
        break;
      case 3:
        // psum DMA
        for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {
          for (size_t images = 0; images < N; images += IMAGES_PER_BURST) {
            size_t buf_offset = 0;

            //bsg_printf("in psum DMA\n");
            // wait until remote psum buffer is ready
            bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (psum_f_N)), 0);
            //bsg_printf(" -- buffer ready\n");

            // brand new psum
            if (!LOAD_PSUM) {
              for (size_t offset = 0; offset < PSUM_BUF_SIZE; offset++) {
                psum_buf_remote[offset] = 0;
              }
            } else {
              // read from previous psum
              for (size_t image_id = 0; image_id < IMAGES_PER_BURST; image_id++) {
                for (size_t filter_id = 0; filter_id < FILTERS_PER_PROCESSING_PASS; filter_id++) {
                  // TODO -- channel
                  for (size_t col = 0; col < Wout; col++) {
                    psum_buf_remote[buf_offset] = omap(image_id+images,filter_id+filters,bsg_x-2,col);
                    buf_offset++;
                  }
                }
              }
            }
            asm volatile("": : :"memory");
            *psum_f_N = 1;
            *psum_f_N_r = 1;
            //bsg_printf(" -- buffer copying done\n");
          }
          // std::cout << " -- end of a pass -- " << std::endl;
        }
        break;
      case 4:
        // compute
        for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {

          //bsg_printf("in compute PE\n");
          // wait until filter buf is filled
          bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (filter_f)), 1);
          //bsg_printf(" -- filter buffer filled\n");

          // pass filter along
          if (PASS_FILTER) {
            //bsg_printf(" -- -- passing filter buffer\n");
            // wait until remote filter buffer is ready
            bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (filter_f_E)), 0);
            //bsg_printf(" -- -- next filter buffer ready\n");
            for (size_t offset = 0; offset < FILTER_BUF_SIZE; offset++) {
              filter_buf_remote[offset] = filter_buf[offset];
            }
            asm volatile("": : :"memory");
            *filter_f_E = 1;
            *filter_f_E_r = 1;
          }

          for (size_t images = 0; images < N; images += IMAGES_PER_BURST) {

            // wait until imap buf is filled
            bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (imap_f)), 1);
            //bsg_printf(" -- imap buffer filled\n");

            // pass imap along
            if (PASS_IMAP) {
              //bsg_printf(" -- -- passing imap buffer\n");
              // wait until remote imap buffer is ready
              bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (imap_f_NE)), 0);
              //bsg_printf(" -- -- next imap buffer ready\n");
              for (size_t offset = 0; offset < IMAP_BUF_SIZE; offset++) {
                imap_buf_remote[offset] = imap_buf[offset];
              }
              asm volatile("": : :"memory");
              *imap_f_NE = 1;
              *imap_f_NE_r = 1;
            }

            // wait until psum buf is filled
            bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (psum_f)), 1);
            //bsg_printf(" -- psum buffer filled\n");

            size_t   imap_offset = 0;
            size_t   psum_offset = 0;
            for (size_t image_id = 0; image_id < IMAGES_PER_BURST; image_id++) {
              size_t filter_offset = 0;
              for (size_t filter_id = 0; filter_id < FILTERS_PER_PROCESSING_PASS; filter_id++) {
                // conv 1d -- just meant to be functional
                // sliding window
                for (size_t window = 0; window < Wout; window++) {
                  // dot product between window and filter
                  for (size_t filter_weight = 0; filter_weight < Wk; filter_weight++) {
                    psum_buf[psum_offset + window] +=
                      imap_buf[imap_offset + window + filter_weight] * filter_buf[filter_offset + filter_weight];
                  }
                }
                psum_offset += Wout;
                filter_offset += Wk;
              }
              imap_offset += Win;
            }

            // signal imap free
            asm volatile("": : :"memory");
            *imap_f = 0;
            *imap_f_SW_r = 0;

            // pass psum along OR write back to global memory
            if (PASS_PSUM) {
              //bsg_printf(" -- -- passing psum buffer\n");
              // wait until remote psum buffer is ready
              bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (psum_f_N)), 0);
              //bsg_printf(" -- -- next psum buffer ready\n");
              for (size_t offset = 0; offset < PSUM_BUF_SIZE; offset++) {
                psum_buf_remote[offset] = psum_buf[offset];
              }
              asm volatile("": : :"memory");
              *psum_f_N = 1;
              *psum_f_N_r = 1;
            } else {
              // write back to omap
              size_t buf_offset = 0;
              for (size_t image_id = 0; image_id < IMAGES_PER_BURST; image_id++) {
                for (size_t filter_id = 0; filter_id < FILTERS_PER_PROCESSING_PASS; filter_id++) {
                  // TODO -- channel
                  for (size_t col = 0; col < Wout; col++) {
                    omap(image_id+images,filter_id+filters,bsg_x-2,col) = psum_buf[buf_offset];
                    buf_offset++;
                  }
                }
              }
            }

            // signal psum and imap free
            asm volatile("": : :"memory");
            *psum_f = 0;
            *psum_f_S_r = 0;
          }
          // signal filter free
          asm volatile("": : :"memory");
          *filter_f = 0;
          *filter_f_W_r = 0;

          // std::cout << " -- end of a pass -- " << std::endl;
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

