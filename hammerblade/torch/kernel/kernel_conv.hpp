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

struct ConvParams{
  uint32_t N;
  uint32_t Cout;
  uint32_t Hout;
  uint32_t Wout;
  uint32_t Cin;
  uint32_t Hin;
  uint32_t Win;
  uint32_t Kh;
  uint32_t Kw;
  uint32_t Sh;
  uint32_t Sw;
  uint32_t Ph;
  uint32_t Pw;

  ConvParams(
    hb_tensor_t* output,
    hb_tensor_t* input,
    hb_tensor_t* weight,
    hb_vector_t* padding,
    hb_vector_t* strides) {
    auto y = HBTensor<float, 4>(output);
    auto x = HBTensor<float, 4>(input);
    auto w = HBTensor<float, 4>(weight);
    auto p = HBVector<uint32_t>(padding);
    auto s = HBVector<uint32_t>(strides);

    // Conv2d parameters
    N = y.dim(0); // number of minibatches
    Cout = y.dim(1); // number of output channels
    Hout = y.dim(2);
    Wout = y.dim(3);
    Cin = x.dim(1); // number of input channels
    Hin = x.dim(2);
    Win = x.dim(3);
    Kh = w.dim(2);
    Kw = w.dim(3);
    Sh = s[0];
    Sw = s[1];
    Ph = p[0];
    Pw = p[1];
  }

  bool compare(
      uint32_t N_, uint32_t Cout_, uint32_t Hout_, uint32_t Wout_,
      uint32_t Cin_, uint32_t Hin_, uint32_t Win_, uint32_t Kh_,
      uint32_t Kw_, uint32_t Sh_, uint32_t Sw_, uint32_t Ph_, uint32_t Pw_) {
    return ((N    == N_   ) && 
            (Cout == Cout_) && 
            (Hout == Hout_) && 
            (Wout == Wout_) && 
            (Cin  == Cin_ ) && 
            (Hin  == Hin_ ) && 
            (Win  == Win_ ) && 
            (Kh   == Kh_  ) && 
            (Kw   == Kw_  ) && 
            (Sh   == Sh_  ) && 
            (Sw   == Sw_  ) && 
            (Ph   == Ph_  ) && 
            (Pw   == Pw_  ));
  }
};

template<uint32_t N, uint32_t Cout, uint32_t Hout, uint32_t Wout,
         uint32_t Cin, uint32_t Hin, uint32_t Win, uint32_t Kh,
         uint32_t Kw, uint32_t Sh = 1, uint32_t Sw = 1, uint32_t Ph = 1,
         uint32_t Pw = 1>
int convolution_forward_template(
          hb_tensor_t* output,
          hb_tensor_t* input,
          hb_tensor_t* weight,
          hb_vector_t* padding,
          hb_vector_t* strides) {
    ConvParams params(output, input, weight, padding, strides);
    if(!params.compare(N, Cout, Hout, Wout,
                       Cin, Hin, Win, Kh,
                       Kw, Sh, Sw, Ph, Pw)) {
      return -1;
    }

    auto y = HBTensor4d<float, N, Cout, Hout, Wout>(output);
    auto x = HBTensor4d<float, N, Cin, Hin, Win>(input);
    auto w = HBTensor4d<float, Cout, Cin, Kh, Kw>(weight);

    // Weights buffer
    register float W_local[KhBufSize][KwBufSize];

    // Circular buffer to hold inputs
    register float X_local[KhBufSize][KwBufSize];

    if(__bsg_id == 0)
      hb_assert_msg(Kh <= KhBufSize && Kw <= KwBufSize,
                    "Conv2d filter doesn't fit in DMEM allocated array");

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    for(uint32_t n = 0; n < N; ++n) {
      for(uint32_t ci = 0; ci < Cin; ++ci) { // input channel first to maximum data reuse
        hb_blocked_for(bsg_tiles_X * bsg_tiles_Y, Cout,
                      [&](size_t co, size_t tg_size_co) {
          // Load the filter w(co, ci, :, :) to dmem
          uint32_t w_offset = w.offset(co, ci, 0, 0);
          auto w_ptr = (__remote float*) w.data_ptr();
          load_weights(W_local, w_ptr, w_offset, Kh, Kw);

          hb_blocked_for(tg_size_co, Hout, [&](size_t yh, size_t tg_size_yh) {
            hb_range yw_range;
            calc_range(&yw_range, Wout, tg_size_yh);
            size_t yw_start = yw_range.start;
            size_t yw_end   = yw_range.end;
            
            // width offset for the accessing local circular buffer
            uint32_t w_off = (Sw * yw_start) % Kw;

            // Load input to local buffer
            for(uint32_t kh = 0; kh < Kh; ++kh) {
              for(uint32_t kw = 0; kw < Kw; ++kw) {
                int32_t xh = Sh * yh - Ph + kh;
                int32_t xw = Sw * yw_start - Pw + kw;

                if(xh >= 0 && xh < Hin && xw >= 0 && xw < Win)
                  X_local[kh][(w_off + kw) % Kw] = x(n, ci, xh, xw);
                else
                  X_local[kh][(w_off + kw) % Kw] = 0.0;
              }
            }

            for(uint32_t yw = yw_start; yw < yw_end; ++yw) {
              w_off = (Sw * yw) % Kw;

              if(yw != yw_start) {
                // Load a new column of input data
                for(uint32_t sw = 0; sw < Sw; ++sw) {
                  int32_t xw = Sw * yw - Pw + Kw - Sw + sw;

                  for(uint32_t kh = 0; kh < Kh; ++kh) {
                    int32_t xh = Sh * yh - Ph + kh;
                    X_local[kh][(Sw * (yw - 1) + sw) % Kw] =
                        (xh >= 0 && xh < Hin && xw >= 0 && xw < Win) ?
                          x(n, ci, xh, xw) : 0;
                  }
                }
              } // y == yw_start would be loaded in the outer loop

              for(uint32_t kh = 0; kh < Kh; ++kh) {
                for(uint32_t kw = 0; kw < Kw; ++kw) {
                  if((ci + kh + kw) == 0) {
                    y(n, co, yh, yw) = 0.0;
                  }
                  y(n, co, yh, yw) += X_local[kh][(w_off + kw) % Kw] *
                                      W_local[kh][kw];
                }
              }
            };
          });
        });
      }
    };

    // End profiling
  bsg_cuda_print_stat_kernel_end();

  g_barrier.sync();
  return 0;
}

}

#endif // _KERNEL_CONV_HPP
