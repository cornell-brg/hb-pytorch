//====================================================================
// Specialized conv1d for SCDA
// Assume ther are 28160 input elements
// so we do a 55 output block
// Assume 1x5 filter
// Assume 2 padding
//====================================================================

#define BLOCK_DIM  55
#define FILTER_DIM  5
#define PADDING     2

#define IMAP_DIM (BLOCK_DIM + FILTER_DIM - 1)

#include <kernel_common.hpp>

namespace {

inline void imapDMA_padding(HBTensor<float, 4>& imap, float* imap_buf,
                            size_t image_id, size_t channel_id,
                            size_t block_id, size_t num_blocks) {
  std::cout << "imapDMA - image id = " << image_id << " channel_id = " << channel_id
            << " block_id = " << block_id << " num_blocks = " << num_blocks << std::endl;
  // helper functions
  auto load2 = [&](float* start_ptr, float* dest_ptr) {
    register float tmp0 = *(start_ptr + 0);
    register float tmp1 = *(start_ptr + 1);
    *(dest_ptr + 0) = tmp0;
    *(dest_ptr + 1) = tmp1;
  };

  auto pad2 = [&](float* dest_ptr) {
    *(dest_ptr + 0) = 0.0f;
    *(dest_ptr + 1) = 0.0f;
  };

  auto load55 = [&](float* start_ptr, float* dest_ptr) {
    // unroll by 11
    for (size_t idx = 0; idx < BLOCK_DIM; idx+=11) {
      register float tmp0  = *(start_ptr + 0);
      register float tmp1  = *(start_ptr + 1);
      register float tmp2  = *(start_ptr + 2);
      register float tmp3  = *(start_ptr + 3);
      register float tmp4  = *(start_ptr + 4);
      register float tmp5  = *(start_ptr + 5);
      register float tmp6  = *(start_ptr + 6);
      register float tmp7  = *(start_ptr + 7);
      register float tmp8  = *(start_ptr + 8);
      register float tmp9  = *(start_ptr + 9);
      register float tmp10 = *(start_ptr + 10);
      asm volatile("": : :"memory");
      *(dest_ptr + 0)  = tmp0;
      *(dest_ptr + 1)  = tmp1;
      *(dest_ptr + 2)  = tmp2;
      *(dest_ptr + 3)  = tmp3;
      *(dest_ptr + 4)  = tmp4;
      *(dest_ptr + 5)  = tmp5;
      *(dest_ptr + 6)  = tmp6;
      *(dest_ptr + 7)  = tmp7;
      *(dest_ptr + 8)  = tmp8;
      *(dest_ptr + 9)  = tmp9;
      *(dest_ptr + 10) = tmp10;
      // advance pointers
      start_ptr += 11;
      dest_ptr  += 11;
    }
  };

  // figure out what we need to do
  // block_id == 0              <- pad left
  // block_id == num_blocks - 1 <- pad right

  float* imap_src_base = (float*)imap.data_ptr();
  const uint32_t* imap_src_strides = imap.get_strides();
  imap_src_base += image_id * imap_src_strides[0] + channel_id * imap_src_strides[1];
  // technically there is a x axis ... but it has to have a stride of 0 in the 1d case
  std::cout << "imap base = " << imap_src_base << " = " << *imap_src_base << std::endl;

  if (block_id == 0) {
    pad2(imap_buf);
    load55(imap_src_base, imap_buf+2);
    load2(imap_src_base+55, imap_buf+57);
  }
  else if (block_id == num_blocks - 1) {
    imap_src_base += (block_id * 55 - 2);
    load2(imap_src_base, imap_buf);
    load55(imap_src_base+2, imap_buf+2);
    pad2(imap_buf+57);
  }
  else {
    imap_src_base += (block_id * 55 - 2);
    load2(imap_src_base, imap_buf);
    load55(imap_src_base+2, imap_buf+2);
    load2(imap_src_base+57, imap_buf+57);
  }

  return;
}

inline void filterDMA(HBTensor<float, 4>& filter, float* filter_buf,
                      size_t filter_id, size_t channel_id) {
  float* filter_src_base = (float*)filter.data_ptr();
  uint32_t* filter_src_strides = filter.get_strides();
  filter_src_base += filter_id * filter_src_strides[0] + channel_id * filter_src_strides[1];
  // filter is known to be 1x5
  float tmp0 = *(filter_src_base + 0);
  float tmp1 = *(filter_src_base + 1);
  float tmp2 = *(filter_src_base + 2);
  float tmp3 = *(filter_src_base + 3);
  float tmp4 = *(filter_src_base + 4);
  asm volatile("": : :"memory");
  *(filter_buf + 0) = tmp0;
  *(filter_buf + 1) = tmp1;
  *(filter_buf + 2) = tmp2;
  *(filter_buf + 3) = tmp3;
  *(filter_buf + 4) = tmp4;
  return;
}

inline void filterDMA_rotate(HBTensor<float, 4>& filter, float* filter_buf,
                      size_t filter_id, size_t channel_id) {
  float* filter_src_base = (float*)filter.data_ptr();
  uint32_t* filter_src_strides = filter.get_strides();
  filter_src_base += filter_id * filter_src_strides[0] + channel_id * filter_src_strides[1];
  // filter is known to be 1x5
  float tmp0 = *(filter_src_base + 0);
  float tmp1 = *(filter_src_base + 1);
  float tmp2 = *(filter_src_base + 2);
  float tmp3 = *(filter_src_base + 3);
  float tmp4 = *(filter_src_base + 4);
  asm volatile("": : :"memory");
  *(filter_buf + 0) = tmp4;
  *(filter_buf + 1) = tmp3;
  *(filter_buf + 2) = tmp2;
  *(filter_buf + 3) = tmp1;
  *(filter_buf + 4) = tmp0;
  return;
}

inline void omapDMA(HBTensor<float, 4>& omap, float* omap_buf,
               size_t image_id, size_t filter_id, size_t block_id) {
  float* omap_src_base = (float*)omap.data_ptr();
  uint32_t* omap_src_strides = omap.get_strides();
  omap_src_base += image_id * omap_src_strides[0] + filter_id * omap_src_strides[1];
  omap_src_base += block_id * BLOCK_DIM;
  // unroll by 11
  for (size_t idx = 0; idx < BLOCK_DIM; idx+=11) {
    *(omap_src_base +  0) = *(omap_buf +  0);
    *(omap_src_base +  1) = *(omap_buf +  1);
    *(omap_src_base +  2) = *(omap_buf +  2);
    *(omap_src_base +  3) = *(omap_buf +  3);
    *(omap_src_base +  4) = *(omap_buf +  4);
    *(omap_src_base +  5) = *(omap_buf +  5);
    *(omap_src_base +  6) = *(omap_buf +  6);
    *(omap_src_base +  7) = *(omap_buf +  7);
    *(omap_src_base +  8) = *(omap_buf +  8);
    *(omap_src_base +  9) = *(omap_buf +  9);
    *(omap_src_base + 10) = *(omap_buf + 10);
    omap_src_base += 11;
    omap_buf      += 11;
  }
  return;
}

inline void omapReset(float* omap_buf) {
  // unroll by 11
  for (size_t idx = 0; idx < BLOCK_DIM; idx+=11) {
    *(omap_buf +  0) = 0;
    *(omap_buf +  1) = 0;
    *(omap_buf +  2) = 0;
    *(omap_buf +  3) = 0;
    *(omap_buf +  4) = 0;
    *(omap_buf +  5) = 0;
    *(omap_buf +  6) = 0;
    *(omap_buf +  7) = 0;
    *(omap_buf +  8) = 0;
    *(omap_buf +  9) = 0;
    *(omap_buf + 10) = 0;
    omap_buf += 11;
  }
  return;
}


} // namespace

extern "C" {

   __attribute__ ((noinline))  int tensorlib_conv1d_forward(
       hb_tensor_t* output,
       hb_tensor_t* input,
       hb_tensor_t* weight,
       hb_vector_t* padding,
       hb_vector_t* strides) {

    HBTensor<float, 4> omap(output);
    HBTensor<float, 4> imap(input);
    HBTensor<float, 4> filter(weight);
    HBVector<uint32_t> p(padding);
    HBVector<uint32_t> s(strides);

    // conv parameters
    auto N    = omap.dim(0); // number of images in batch
    auto Cout = omap.dim(1); // number of output channels
    auto Hout = omap.dim(2);
    auto Wout = omap.dim(3);
    auto Cin  = imap.dim(1); // number of input channels
    auto Hin  = imap.dim(2);
    auto Win  = imap.dim(3);
    auto Hk   = filter.dim(2);
    auto Wk   = filter.dim(3);

    hb_assert(Hk == 1);
    hb_assert(Wk == FILTER_DIM);
    hb_assert(p[0] == 0);
    hb_assert(p[1] == PADDING);
    hb_assert(s[0] == 1);
    hb_assert(s[1] == 1);
    hb_assert(Wout % BLOCK_DIM == 0);

    size_t blocks_per_out_channel = Wout / BLOCK_DIM;
    size_t num_blocks = N * Cout * blocks_per_out_channel;

    // allocate buffers
    float filter_buf[FILTER_DIM];
    float omap_buf[BLOCK_DIM];
    float imap_buf[IMAP_DIM];

     bsg_cuda_print_stat_start(4);

     for (size_t idx = bsg_id; idx < num_blocks; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
       size_t tmp = idx;
       size_t image_id = tmp / (Cout * blocks_per_out_channel);
       tmp = tmp % (Cout * blocks_per_out_channel);
       size_t filter_id = tmp / blocks_per_out_channel;
       tmp = tmp % blocks_per_out_channel;
       size_t block_id = tmp % blocks_per_out_channel;

       omapReset(omap_buf);

       for (size_t channel_id = 0; channel_id < Cin; channel_id++) {
         imapDMA_padding(imap, imap_buf, image_id, channel_id, block_id, num_blocks);
         filterDMA(filter, filter_buf, filter_id, channel_id);
         // debug print
         for (size_t i = 0; i < IMAP_DIM; i++) {
           std::cout << imap_buf[i] << ",";
         }
         std::cout << std::endl;
         // do conv
       } //channel
       omapDMA(omap, omap_buf, image_id, filter_id, block_id);
     }

     bsg_cuda_print_stat_end(4);

    g_barrier.sync();
    return 0;
   }

   HB_EMUL_REG_KERNEL(tensorlib_conv1d_forward, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                      hb_vector_t*, hb_vector_t*)
}
