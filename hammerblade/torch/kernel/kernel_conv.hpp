//====================================================================
// Convolution kernel
// 27/07/2020 Bandhav Veluri
//====================================================================

#ifndef _KERNEL_CONV_HPP
#define _KERNEL_CONV_HPP

namespace {

// Size of buffers allocated for filters in DMEM
const uint32_t KhBufSize = 5;
const uint32_t KwBufSize = 5;

inline void load_weights(float wl[KhBufSize][KwBufSize],
                         __remote float* NOALIAS wr,
                         uint32_t offset, uint32_t Kh, uint32_t Kw) {
  for(int i = 0; i < Kh; ++i) {
    for(int j = 0; j < Kw; ++j) {
      wl[i][j] = wr[offset + i * Kw + j];
    }
  }
}

}

#endif // _KERNEL_CONV_HPP
