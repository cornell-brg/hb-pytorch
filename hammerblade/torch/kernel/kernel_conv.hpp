//====================================================================
// Convolution kernel
// 27/07/2020 Bandhav Veluri
//====================================================================

struct Conv2dParams {
  uint32_t N;
  uint32_t Cout
  uint32_t Hout
  uint32_t Wout
  uint32_t Cin;
  uint32_t Hin;
  uint32_t Win;
  uint32_t Kh;
  uint32_t Kw;
  uint32_t Sh;
  uint32_t Sw;
  uint32_t Ph;
  uint32_t Pw;

  Conv2dParams(HBTensor<float> x,
               HBTensor<float> y,
               HBTensor<float> w,
               HBVector<float> s,
               HBVector<float> p) {
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
};
