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

inline void kernel1d(float* imap, float* filter, float* omap) {
  // unroll by 11
  // 11 to hold output
  //  5 to hold filter
  // 15 to hold input
  // 11 + 5 + 15 = 31 < 32
  for (size_t idx = 0; idx < BLOCK_DIM; idx+=11) {
    register float psum0  = omap[idx +  0];
    register float psum1  = omap[idx +  1];
    register float psum2  = omap[idx +  2];
    register float psum3  = omap[idx +  3];
    register float psum4  = omap[idx +  4];
    register float psum5  = omap[idx +  5];
    register float psum6  = omap[idx +  6];
    register float psum7  = omap[idx +  7];
    register float psum8  = omap[idx +  8];
    register float psum9  = omap[idx +  9];
    register float psum10 = omap[idx + 10];

    register float filter0 = filter[0];
    register float filter1 = filter[1];
    register float filter2 = filter[2];
    register float filter3 = filter[3];
    register float filter4 = filter[4];

    register float imap0  = imap[idx +  0];
    register float imap1  = imap[idx +  1];
    register float imap2  = imap[idx +  2];
    register float imap3  = imap[idx +  3];
    register float imap4  = imap[idx +  4];
    register float imap5  = imap[idx +  5];
    register float imap6  = imap[idx +  6];
    register float imap7  = imap[idx +  7];
    register float imap8  = imap[idx +  8];
    register float imap9  = imap[idx +  9];
    register float imap10 = imap[idx + 10];
    register float imap11 = imap[idx + 11];
    register float imap12 = imap[idx + 12];
    register float imap13 = imap[idx + 13];
    register float imap14 = imap[idx + 14];
    asm volatile("": : :"memory");

#ifdef HB_EMUL
    psum0  += imap0  * filter0;
    psum1  += imap1  * filter0;
    psum2  += imap2  * filter0;
    psum3  += imap3  * filter0;
    psum4  += imap4  * filter0;
    psum5  += imap5  * filter0;
    psum6  += imap6  * filter0;
    psum7  += imap7  * filter0;
    psum8  += imap8  * filter0;
    psum9  += imap9  * filter0;
    psum10 += imap10 * filter0;

    psum0  += imap1  * filter1;
    psum1  += imap2  * filter1;
    psum2  += imap3  * filter1;
    psum3  += imap4  * filter1;
    psum4  += imap5  * filter1;
    psum5  += imap6  * filter1;
    psum6  += imap7  * filter1;
    psum7  += imap8  * filter1;
    psum8  += imap9  * filter1;
    psum9  += imap10 * filter1;
    psum10 += imap11 * filter1;

    psum0  += imap2  * filter2;
    psum1  += imap3  * filter2;
    psum2  += imap4  * filter2;
    psum3  += imap5  * filter2;
    psum4  += imap6  * filter2;
    psum5  += imap7  * filter2;
    psum6  += imap8  * filter2;
    psum7  += imap9  * filter2;
    psum8  += imap10 * filter2;
    psum9  += imap11 * filter2;
    psum10 += imap12 * filter2;

    psum0  += imap3  * filter3;
    psum1  += imap4  * filter3;
    psum2  += imap5  * filter3;
    psum3  += imap6  * filter3;
    psum4  += imap7  * filter3;
    psum5  += imap8  * filter3;
    psum6  += imap9  * filter3;
    psum7  += imap10 * filter3;
    psum8  += imap11 * filter3;
    psum9  += imap12 * filter3;
    psum10 += imap13 * filter3;

    psum0  += imap4  * filter4;
    psum1  += imap5  * filter4;
    psum2  += imap6  * filter4;
    psum3  += imap7  * filter4;
    psum4  += imap8  * filter4;
    psum5  += imap9  * filter4;
    psum6  += imap10 * filter4;
    psum7  += imap11 * filter4;
    psum8  += imap12 * filter4;
    psum9  += imap13 * filter4;
    psum10 += imap14 * filter4;
#else
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum0)  : "f"(imap0),  "f"(filter0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum1)  : "f"(imap1),  "f"(filter0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum2)  : "f"(imap2),  "f"(filter0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum3)  : "f"(imap3),  "f"(filter0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum4)  : "f"(imap4),  "f"(filter0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum5)  : "f"(imap5),  "f"(filter0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum6)  : "f"(imap6),  "f"(filter0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum7)  : "f"(imap7),  "f"(filter0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum8)  : "f"(imap8),  "f"(filter0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum9)  : "f"(imap9),  "f"(filter0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum10) : "f"(imap10), "f"(filter0));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum0)  : "f"(imap1),  "f"(filter1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum1)  : "f"(imap2),  "f"(filter1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum2)  : "f"(imap3),  "f"(filter1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum3)  : "f"(imap4),  "f"(filter1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum4)  : "f"(imap5),  "f"(filter1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum5)  : "f"(imap6),  "f"(filter1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum6)  : "f"(imap7),  "f"(filter1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum7)  : "f"(imap8),  "f"(filter1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum8)  : "f"(imap9),  "f"(filter1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum9)  : "f"(imap10), "f"(filter1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum10) : "f"(imap11), "f"(filter1));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum0)  : "f"(imap2),  "f"(filter2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum1)  : "f"(imap3),  "f"(filter2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum2)  : "f"(imap4),  "f"(filter2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum3)  : "f"(imap5),  "f"(filter2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum4)  : "f"(imap6),  "f"(filter2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum5)  : "f"(imap7),  "f"(filter2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum6)  : "f"(imap8),  "f"(filter2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum7)  : "f"(imap9),  "f"(filter2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum8)  : "f"(imap10), "f"(filter2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum9)  : "f"(imap11), "f"(filter2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum10) : "f"(imap12), "f"(filter2));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum0)  : "f"(imap3),  "f"(filter3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum1)  : "f"(imap4),  "f"(filter3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum2)  : "f"(imap5),  "f"(filter3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum3)  : "f"(imap6),  "f"(filter3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum4)  : "f"(imap7),  "f"(filter3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum5)  : "f"(imap8),  "f"(filter3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum6)  : "f"(imap9),  "f"(filter3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum7)  : "f"(imap10), "f"(filter3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum8)  : "f"(imap11), "f"(filter3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum9)  : "f"(imap12), "f"(filter3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum10) : "f"(imap13), "f"(filter3));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum0)  : "f"(imap4),  "f"(filter4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum1)  : "f"(imap5),  "f"(filter4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum2)  : "f"(imap6),  "f"(filter4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum3)  : "f"(imap7),  "f"(filter4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum4)  : "f"(imap8),  "f"(filter4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum5)  : "f"(imap9),  "f"(filter4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum6)  : "f"(imap10), "f"(filter4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum7)  : "f"(imap11), "f"(filter4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum8)  : "f"(imap12), "f"(filter4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum9)  : "f"(imap13), "f"(filter4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum10) : "f"(imap14), "f"(filter4));
#endif
    omap[idx +  0] = psum0;
    omap[idx +  1] = psum1;
    omap[idx +  2] = psum2;
    omap[idx +  3] = psum3;
    omap[idx +  4] = psum4;
    omap[idx +  5] = psum5;
    omap[idx +  6] = psum6;
    omap[idx +  7] = psum7;
    omap[idx +  8] = psum8;
    omap[idx +  9] = psum9;
    omap[idx + 10] = psum10;
  }

  return;
}

inline void gradDMA(HBTensor<float, 4>& grad, float* grad_buf,
                    size_t image_id, size_t filter_id,
                    size_t block_id) {

  float* grad_src_base = (float*)grad.data_ptr();
  uint32_t* grad_src_strides = grad.get_strides();
  grad_src_base += image_id * grad_src_strides[0] + filter_id * grad_src_strides[1];
  grad_src_base += block_id * BLOCK_DIM;
  // unroll by 11
  for (size_t idx = 0; idx < BLOCK_DIM; idx+=11) {
    register float tmp0  = *(grad_src_base + 0);
    register float tmp1  = *(grad_src_base + 1);
    register float tmp2  = *(grad_src_base + 2);
    register float tmp3  = *(grad_src_base + 3);
    register float tmp4  = *(grad_src_base + 4);
    register float tmp5  = *(grad_src_base + 5);
    register float tmp6  = *(grad_src_base + 6);
    register float tmp7  = *(grad_src_base + 7);
    register float tmp8  = *(grad_src_base + 8);
    register float tmp9  = *(grad_src_base + 9);
    register float tmp10 = *(grad_src_base + 10);
    asm volatile("": : :"memory");
    *(grad_buf + 0)  = tmp0;
    *(grad_buf + 1)  = tmp1;
    *(grad_buf + 2)  = tmp2;
    *(grad_buf + 3)  = tmp3;
    *(grad_buf + 4)  = tmp4;
    *(grad_buf + 5)  = tmp5;
    *(grad_buf + 6)  = tmp6;
    *(grad_buf + 7)  = tmp7;
    *(grad_buf + 8)  = tmp8;
    *(grad_buf + 9)  = tmp9;
    *(grad_buf + 10) = tmp10;
    // advance pointers
    grad_src_base += 11;
    grad_buf      += 11;
  }
  return;
}


inline void kernel1d_back_weight(float* imap, float* filter, float* grad) {
  // load partial filter
  register float filter0 = filter[0];
  register float filter1 = filter[1];
  register float filter2 = filter[2];
  register float filter3 = filter[3];
  register float filter4 = filter[4];

  for (size_t idx = 0; idx < BLOCK_DIM; idx+=11) {
    register float imap0  = imap[idx +  0];
    register float imap1  = imap[idx +  1];
    register float imap2  = imap[idx +  2];
    register float imap3  = imap[idx +  3];
    register float imap4  = imap[idx +  4];
    register float imap5  = imap[idx +  5];
    register float imap6  = imap[idx +  6];
    register float imap7  = imap[idx +  7];
    register float imap8  = imap[idx +  8];
    register float imap9  = imap[idx +  9];
    register float imap10 = imap[idx + 10];
    register float imap11 = imap[idx + 11];
    register float imap12 = imap[idx + 12];
    register float imap13 = imap[idx + 13];
    register float imap14 = imap[idx + 14];

    register float grad0  = grad[idx +  0];
    register float grad1  = grad[idx +  1];
    register float grad2  = grad[idx +  2];
    register float grad3  = grad[idx +  3];
    register float grad4  = grad[idx +  4];
    register float grad5  = grad[idx +  5];
    register float grad6  = grad[idx +  6];
    register float grad7  = grad[idx +  7];
    register float grad8  = grad[idx +  8];
    register float grad9  = grad[idx +  9];
    register float grad10 = grad[idx + 10];
    asm volatile("": : :"memory");

#ifdef HB_EMUL
    filter0 += imap0  * grad0;
    filter1 += imap1  * grad0;
    filter2 += imap2  * grad0;
    filter3 += imap3  * grad0;
    filter4 += imap4  * grad0;

    filter0 += imap1  * grad1;
    filter1 += imap2  * grad1;
    filter2 += imap3  * grad1;
    filter3 += imap4  * grad1;
    filter4 += imap5  * grad1;

    filter0 += imap2  * grad2;
    filter1 += imap3  * grad2;
    filter2 += imap4  * grad2;
    filter3 += imap5  * grad2;
    filter4 += imap6  * grad2;

    filter0 += imap3  * grad3;
    filter1 += imap4  * grad3;
    filter2 += imap5  * grad3;
    filter3 += imap6  * grad3;
    filter4 += imap7  * grad3;

    filter0 += imap4  * grad4;
    filter1 += imap5  * grad4;
    filter2 += imap6  * grad4;
    filter3 += imap7  * grad4;
    filter4 += imap8  * grad4;

    filter0 += imap5  * grad5;
    filter1 += imap6  * grad5;
    filter2 += imap7  * grad5;
    filter3 += imap8  * grad5;
    filter4 += imap9  * grad5;

    filter0 += imap6  * grad6;
    filter1 += imap7  * grad6;
    filter2 += imap8  * grad6;
    filter3 += imap9  * grad6;
    filter4 += imap10 * grad6;

    filter0 += imap7  * grad7;
    filter1 += imap8  * grad7;
    filter2 += imap9  * grad7;
    filter3 += imap10 * grad7;
    filter4 += imap11 * grad7;

    filter0 += imap8  * grad8;
    filter1 += imap9  * grad8;
    filter2 += imap10 * grad8;
    filter3 += imap11 * grad8;
    filter4 += imap12 * grad8;

    filter0 += imap9  * grad9;
    filter1 += imap10 * grad9;
    filter2 += imap11 * grad9;
    filter3 += imap12 * grad9;
    filter4 += imap13 * grad9;

    filter0 += imap10 * grad10;
    filter1 += imap11 * grad10;
    filter2 += imap12 * grad10;
    filter3 += imap13 * grad10;
    filter4 += imap14 * grad10;
#else
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter0)  : "f"(imap0),  "f"(grad0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter1)  : "f"(imap1),  "f"(grad0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter2)  : "f"(imap2),  "f"(grad0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter3)  : "f"(imap3),  "f"(grad0));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter4)  : "f"(imap4),  "f"(grad0));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter0)  : "f"(imap1),  "f"(grad1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter1)  : "f"(imap2),  "f"(grad1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter2)  : "f"(imap3),  "f"(grad1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter3)  : "f"(imap4),  "f"(grad1));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter4)  : "f"(imap5),  "f"(grad1));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter0)  : "f"(imap2),  "f"(grad2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter1)  : "f"(imap3),  "f"(grad2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter2)  : "f"(imap4),  "f"(grad2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter3)  : "f"(imap5),  "f"(grad2));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter4)  : "f"(imap6),  "f"(grad2));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter0)  : "f"(imap3),  "f"(grad3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter1)  : "f"(imap4),  "f"(grad3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter2)  : "f"(imap5),  "f"(grad3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter3)  : "f"(imap6),  "f"(grad3));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter4)  : "f"(imap7),  "f"(grad3));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter0)  : "f"(imap4),  "f"(grad4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter1)  : "f"(imap5),  "f"(grad4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter2)  : "f"(imap6),  "f"(grad4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter3)  : "f"(imap7),  "f"(grad4));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter4)  : "f"(imap8),  "f"(grad4));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter0)  : "f"(imap5),  "f"(grad5));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter1)  : "f"(imap6),  "f"(grad5));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter2)  : "f"(imap7),  "f"(grad5));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter3)  : "f"(imap8),  "f"(grad5));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter4)  : "f"(imap9),  "f"(grad5));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter0)  : "f"(imap6),  "f"(grad6));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter1)  : "f"(imap7),  "f"(grad6));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter2)  : "f"(imap8),  "f"(grad6));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter3)  : "f"(imap9),  "f"(grad6));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter4)  : "f"(imap10), "f"(grad6));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter0)  : "f"(imap7),  "f"(grad7));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter1)  : "f"(imap8),  "f"(grad7));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter2)  : "f"(imap9),  "f"(grad7));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter3)  : "f"(imap10), "f"(grad7));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter4)  : "f"(imap11), "f"(grad7));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter0)  : "f"(imap8),  "f"(grad8));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter1)  : "f"(imap9),  "f"(grad8));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter2)  : "f"(imap10), "f"(grad8));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter3)  : "f"(imap11), "f"(grad8));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter4)  : "f"(imap12), "f"(grad8));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter0)  : "f"(imap9),  "f"(grad9));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter1)  : "f"(imap10), "f"(grad9));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter2)  : "f"(imap11), "f"(grad9));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter3)  : "f"(imap12), "f"(grad9));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter4)  : "f"(imap13), "f"(grad9));

    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter0)  : "f"(imap10), "f"(grad10));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter1)  : "f"(imap11), "f"(grad10));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter2)  : "f"(imap12), "f"(grad10));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter3)  : "f"(imap13), "f"(grad10));
    asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(filter4)  : "f"(imap14), "f"(grad10));
#endif
  }
    filter[0] = filter0;
    filter[1] = filter1;
    filter[2] = filter2;
    filter[3] = filter3;
    filter[4] = filter4;
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
    hb_assert(Hin == 1);
    hb_assert(Hout == 1);
    hb_assert(Wout % BLOCK_DIM == 0);

    size_t blocks_per_out_channel = Wout / BLOCK_DIM;
    size_t num_blocks = N * blocks_per_out_channel;

    // allocate buffers
    float filter_buf[FILTER_DIM];
    float omap_buf[BLOCK_DIM];
    float imap_buf[IMAP_DIM*8];

    // SM Hack
    size_t tiles_to_hold_full_input = (Cin + 15) / 16;
    size_t filters_per_tile = (Cout + tiles_to_hold_full_input - 1) / tiles_to_hold_full_input;
    size_t tile_offset = bsg_id % tiles_to_hold_full_input;

    bsg_cuda_print_stat_start(4);

    for (size_t idx = bsg_id; idx < num_blocks; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
      size_t tmp = idx;
      size_t image_id = tmp / blocks_per_out_channel;
      size_t block_id = tmp % blocks_per_out_channel;

      // load input
      size_t start = tile_offset;
      size_t end = (start + 16) > Cin ? Cin :  (start + 16);
      for (size_t c = start; c < end; c++) {
        imapDMA_padding(imap, imap_buf+c, image_id, c, block_id, blocks_per_out_channel);
      }

      // re-use on all filters
      start = tile_offset;
      end = (tile_offset + filters_per_tile) > Cout ? Cout : (tile_offset + filters_per_tile);
      for (size_t filter_id = start; filter_id < end; filter_id++) {
        omapReset(omap_buf);

        for (size_t channel_id = 0; channel_id < Cin; channel_id++) {
          filterDMA(filter, filter_buf, filter_id, channel_id);
          // do conv
          kernel1d(imap_buf, filter_buf, omap_buf);
        } //channel
        omapDMA(omap, omap_buf, image_id, filter_id, block_id);
      }

      // early quit
      break;
    }

    bsg_cuda_print_stat_end(4);

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_conv1d_forward, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                      hb_vector_t*, hb_vector_t*)


  __attribute__ ((noinline)) int tensorlib_conv1d_backward_input(
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
    hb_assert(Hin == 1);
    hb_assert(Hout == 1);
    hb_assert(Wout % BLOCK_DIM == 0);

    size_t blocks_per_out_channel = Wout / BLOCK_DIM;
    size_t num_blocks = N * blocks_per_out_channel;

    // allocate buffers
    float filter_buf[FILTER_DIM];
    float omap_buf[BLOCK_DIM];
    float imap_buf[IMAP_DIM*8];

    // SM Hack
    size_t tiles_to_hold_full_input = (Cin + 15) / 16;
    size_t channels_per_tile = (Cout + tiles_to_hold_full_input - 1) / tiles_to_hold_full_input;
    size_t tile_offset = bsg_id % tiles_to_hold_full_input;

    bsg_cuda_print_stat_start(5);

    for (size_t idx = bsg_id; idx < num_blocks; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
      size_t tmp = idx;
      size_t image_id = tmp / blocks_per_out_channel;
      size_t block_id = tmp % blocks_per_out_channel;

      // load input
      size_t start = tile_offset;
      size_t end = (start + 16) > Cin ? Cin :  (start + 16);
      for (size_t f = start; f < end; f++) {
        imapDMA_padding(imap, imap_buf+f, image_id, f, block_id, blocks_per_out_channel);
      }

      // reuse input
      start = tile_offset;
      end = (tile_offset + channels_per_tile) > Cout ? Cout : (tile_offset + channels_per_tile);
      for (size_t channel_id = start; channel_id < end; channel_id++) {
        omapReset(omap_buf);

        for (size_t filter_id = 0; filter_id < Cin; filter_id++) {
          filterDMA_rotate(filter, filter_buf, filter_id, channel_id);
          kernel1d(imap_buf, filter_buf, omap_buf);
        }
        omapDMA(omap, omap_buf, image_id, channel_id, block_id);
      }
      // early quit
      break;
    }

    bsg_cuda_print_stat_end(5);

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_conv1d_backward_input, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_vector_t*, hb_vector_t*)

  __attribute__ ((noinline)) int tensorlib_conv1d_backward_weight(
      hb_tensor_t* output,
      hb_tensor_t* grad_p,
      hb_tensor_t* imap_p,
      hb_vector_t* padding,
      hb_vector_t* strides) {

    HBTensor<float, 4> filter(output);
    HBTensor<float, 4> grad(grad_p);
    HBTensor<float, 4> imap(imap_p);
    HBVector<uint32_t> p(padding);
    HBVector<uint32_t> s(strides);

    // conv parameters
    auto N      = filter.dim(0); // number of filters to calculate grad for
    auto Cout   = filter.dim(1); // number of channels in the images
    auto Hout   = filter.dim(2);
    auto Wout   = filter.dim(3); // filter dimensions
    auto N_imap = imap.dim(0);   // number of images
    auto Hin    = imap.dim(2);   // image dimensions
    auto Win    = imap.dim(3);
    auto Hk     = grad.dim(2);   // grad dimensions
    auto Wk     = grad.dim(3);

    hb_assert(Hout == 1);
    hb_assert(Wout == FILTER_DIM);
    hb_assert(p[0] == 0);
    hb_assert(p[1] == PADDING);
    hb_assert(s[0] == 1);
    hb_assert(s[1] == 1);
    hb_assert(Hin == 1);
    hb_assert(Hout == 1);
    hb_assert(Hk == 1);
    hb_assert(Win % BLOCK_DIM == 0);
    hb_assert(Wk % BLOCK_DIM == 0);

    size_t blocks_per_out_channel = Wk / BLOCK_DIM;
    size_t num_blocks = N * Cout; // parallel over filter x channel

    float filter_buf[FILTER_DIM];
    float imap_buf[IMAP_DIM];
    float grad_buf[BLOCK_DIM];

    bsg_cuda_print_stat_start(6);

    for (size_t idx = bsg_id; idx < num_blocks; idx += (BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM)) {
      size_t filter_id = idx / Cout;
      size_t channel_id = idx % Cout;

      // reset filter buf
      filter_buf[0] = 0;
      filter_buf[1] = 0;
      filter_buf[2] = 0;
      filter_buf[3] = 0;
      filter_buf[4] = 0;

      for (size_t image_id = 0; image_id < N_imap; image_id++) {
        for (size_t block_id = 0; block_id < blocks_per_out_channel; block_id++) {

          imapDMA_padding(imap, imap_buf, image_id, channel_id, block_id, blocks_per_out_channel);
          gradDMA(grad, grad_buf, image_id, filter_id, block_id);
          // do large conv
          kernel1d_back_weight(imap_buf, filter_buf, grad_buf);
        }
      }
      // filter WB
      bsg_attr_remote float* filter_dst_base = (float*)filter.data_ptr();
      uint32_t* filter_dst_strides = filter.get_strides();
      filter_dst_base += filter_id * filter_dst_strides[0] + channel_id * filter_dst_strides[1];
      *(filter_dst_base + 0) = filter_buf[0];
      *(filter_dst_base + 1) = filter_buf[1];
      *(filter_dst_base + 2) = filter_buf[2];
      *(filter_dst_base + 3) = filter_buf[3];
      *(filter_dst_base + 4) = filter_buf[4];
      break;
    }

    bsg_cuda_print_stat_end(6);

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_conv1d_backward_weight, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_vector_t*, hb_vector_t*)
}
