//====================================================================
// Vector - vector add kernel
// 06/02/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

template<typename scalar_t>
inline uint32_t offset(uint32_t idx, HBTensor<scalar_t> tensor) {
  uint32_t* strides = tensor.get_strides();
  uint32_t* sizes = tensor.get_sizes();
  uint32_t offset = 0;
  for(uint32_t i = 0; i < tensor.ndim(); i++) {
    uint32_t dimx = idx % sizes[i];
    idx /= sizes[i];
    offset += dimx * strides[i];
  }
  return offset;
}

template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll13_try(HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
                               F functor) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t len_per_tile = result.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > result.numel())  ? result.numel() : end;

  // is_trivial_1d
  if(result.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (result.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];

    size_t i = start;
    while ( i + 12 < end){
      scalar_t x1 = *(data[1] + i * strides[1]);
      scalar_t y1 = *(data[2] + i * strides[2]);
      scalar_t x2 = *(data[1] + (i+1) * strides[1]);
      scalar_t y2 = *(data[2] + (i+1) * strides[2]);
      scalar_t x3 = *(data[1] + (i+2) * strides[1]);
      scalar_t y3 = *(data[2] + (i+2) * strides[2]);
      scalar_t x4 = *(data[1] + (i+3) * strides[1]);
      scalar_t y4 = *(data[2] + (i+3) * strides[2]);
      scalar_t x5 = *(data[1] + (i+4) * strides[1]);
      scalar_t y5 = *(data[2] + (i+4) * strides[2]);
      scalar_t x6 = *(data[1] + (i+5) * strides[1]);
      scalar_t y6 = *(data[2] + (i+5) * strides[2]);
      scalar_t x7 = *(data[1] + (i+6) * strides[1]);
      scalar_t y7 = *(data[2] + (i+6) * strides[2]);
      scalar_t x8 = *(data[1] + (i+7) * strides[1]);
      scalar_t y8 = *(data[2] + (i+7) * strides[2]);
      scalar_t x9 = *(data[1] + (i+8) * strides[1]);
      scalar_t y9 = *(data[2] + (i+8) * strides[2]);
      scalar_t x10 = *(data[1] + (i+9) * strides[1]);
      scalar_t y10 = *(data[2] + (i+9) * strides[2]);
      scalar_t x11 = *(data[1] + (i+10) * strides[1]);
      scalar_t y11 = *(data[2] + (i+10) * strides[2]);
      scalar_t x12 = *(data[1] + (i+11) * strides[1]);
      scalar_t y12 = *(data[2] + (i+11) * strides[2]);
      scalar_t x13 = *(data[1] + (i+12) * strides[1]);
      scalar_t y13 = *(data[2] + (i+12) * strides[2]);

      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      scalar_t res4 = functor(x4, y4);
      scalar_t res5 = functor(x5, y5);
      scalar_t res6 = functor(x6, y6);
      scalar_t res7 = functor(x7, y7);
      scalar_t res8 = functor(x8, y8);
      scalar_t res9 = functor(x9, y9);
      scalar_t res10 = functor(x10, y10);
      scalar_t res11 = functor(x11, y11);
      scalar_t res12 = functor(x12, y12);
      scalar_t res13 = functor(x13, y13);

      *(data[0] + strides[0]*i++) = res1;
      *(data[0] + strides[0]*i++) = res2;
      *(data[0] + strides[0]*i++) = res3;
      *(data[0] + strides[0]*i++) = res4;
      *(data[0] + strides[0]*i++) = res5;
      *(data[0] + strides[0]*i++) = res6;
      *(data[0] + strides[0]*i++) = res7;
      *(data[0] + strides[0]*i++) = res8;
      *(data[0] + strides[0]*i++) = res9;
      *(data[0] + strides[0]*i++) = res10;
      *(data[0] + strides[0]*i++) = res11;
      *(data[0] + strides[0]*i++) = res12;
      *(data[0] + strides[0]*i++) = res13;
    }
    if (start + 13 < end) {
      i -= 13;
    }
    while (i < end) {
      scalar_t x = *(data[1] + i * strides[1]);
      scalar_t y = *(data[2] + i * strides[2]);
      scalar_t res = x + y;
      *(data[0] + strides[0]*i++) = res;
    }
  }
  else {
    size_t i = start;
    while ( i + 12 < end){
      scalar_t x1 = *(data[1] + offset(i, input));
      scalar_t y1 = *(data[2] + offset(i, other));
      scalar_t x2 = *(data[1] + offset(i+1, input));
      scalar_t y2 = *(data[2] + offset(i+1, other));
      scalar_t x3 = *(data[1] + offset(i+2, input));
      scalar_t y3 = *(data[2] + offset(i+2, other));
      scalar_t x4 = *(data[1] + offset(i+3, input));
      scalar_t y4 = *(data[2] + offset(i+3, other));
      scalar_t x5 = *(data[1] + offset(i+4, input));
      scalar_t y5 = *(data[2] + offset(i+4, other));
      scalar_t x6 = *(data[1] + offset(i+5, input));
      scalar_t y6 = *(data[2] + offset(i+5, other));
      scalar_t x7 = *(data[1] + offset(i+6, input));
      scalar_t y7 = *(data[2] + offset(i+6, other));
      scalar_t x8 = *(data[1] + offset(i+7, input));
      scalar_t y8 = *(data[2] + offset(i+7, other));
      scalar_t x9 = *(data[1] + offset(i+8, input));
      scalar_t y9 = *(data[2] + offset(i+8, other));
      scalar_t x10 = *(data[1] + offset(i+9, input));
      scalar_t y10 = *(data[2] + offset(i+9, other));
      scalar_t x11 = *(data[1] + offset(i+10, input));
      scalar_t y11 = *(data[2] + offset(i+10, other));
      scalar_t x12 = *(data[1] + offset(i+11, other));
      scalar_t y12 = *(data[2] + offset(i+11, other));
      scalar_t x13 = *(data[1] + offset(i+12, other));
      scalar_t y13 = *(data[2] + offset(i+12, other));

      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      scalar_t res4 = functor(x4, y4);
      scalar_t res5 = functor(x5, y5);
      scalar_t res6 = functor(x6, y6);
      scalar_t res7 = functor(x7, y7);
      scalar_t res8 = functor(x8, y8);
      scalar_t res9 = functor(x9, y9);
      scalar_t res10 = functor(x10, y10);
      scalar_t res11 = functor(x11, y11);
      scalar_t res12 = functor(x12, y12);
      scalar_t res13 = functor(x13, y13);

      *(data[0] + offset(i, result)) = res1;
      *(data[0] + offset(i+1, result)) = res2;
      *(data[0] + offset(i+2, result)) = res3;
      *(data[0] + offset(i+3, result)) = res4;
      *(data[0] + offset(i+4, result)) = res5;
      *(data[0] + offset(i+5, result)) = res6;
      *(data[0] + offset(i+6, result)) = res7;
      *(data[0] + offset(i+7, result)) = res8;
      *(data[0] + offset(i+8, result)) = res9;
      *(data[0] + offset(i+9, result)) = res10;
      *(data[0] + offset(i+10, result)) = res11;
      *(data[0] + offset(i+11, result)) = res12;
      *(data[0] + offset(i+12, result)) = res13;
      i += 13;
    }
    if (start + 13 < end) {
      i -= 13;
    }
    while (i < end) {
      scalar_t x = *(data[1] + offset(i, input));
      scalar_t y = *(data[2] + offset(i, other));
      scalar_t res = functor(x, y);
      *(data[0] + offset(i, result)) = res;
      i++;
    }
  } 
}


template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll11_try(HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
                               F functor) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();


  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t len_per_tile = result.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > result.numel())  ? result.numel() : end;

  // is_trivial_1d
  if(result.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (result.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];

    size_t i = start;
    while ( i + 10 < end){
      scalar_t x1 = *(data[1] + i * strides[1]);
      scalar_t y1 = *(data[2] + i * strides[2]);
      scalar_t x2 = *(data[1] + (i+1) * strides[1]);
      scalar_t y2 = *(data[2] + (i+1) * strides[2]);
      scalar_t x3 = *(data[1] + (i+2) * strides[1]);
      scalar_t y3 = *(data[2] + (i+2) * strides[2]);
      scalar_t x4 = *(data[1] + (i+3) * strides[1]);
      scalar_t y4 = *(data[2] + (i+3) * strides[2]);
      scalar_t x5 = *(data[1] + (i+4) * strides[1]);
      scalar_t y5 = *(data[2] + (i+4) * strides[2]);
      scalar_t x6 = *(data[1] + (i+5) * strides[1]);
      scalar_t y6 = *(data[2] + (i+5) * strides[2]);
      scalar_t x7 = *(data[1] + (i+6) * strides[1]);
      scalar_t y7 = *(data[2] + (i+6) * strides[2]);
      scalar_t x8 = *(data[1] + (i+7) * strides[1]);
      scalar_t y8 = *(data[2] + (i+7) * strides[2]);
      scalar_t x9 = *(data[1] + (i+8) * strides[1]);
      scalar_t y9 = *(data[2] + (i+8) * strides[2]);
      scalar_t x10 = *(data[1] + (i+9) * strides[1]);
      scalar_t y10 = *(data[2] + (i+9) * strides[2]);
      scalar_t x11 = *(data[1] + (i+10) * strides[1]);
      scalar_t y11 = *(data[2] + (i+10) * strides[2]);

      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      scalar_t res4 = functor(x4, y4);
      scalar_t res5 = functor(x5, y5);
      scalar_t res6 = functor(x6, y6);
      scalar_t res7 = functor(x7, y7);
      scalar_t res8 = functor(x8, y8);
      scalar_t res9 = functor(x9, y9);
      scalar_t res10 = functor(x10, y10);
      scalar_t res11 = functor(x11, y11);

      *(data[0] + strides[0]*i++) = res1;
      *(data[0] + strides[0]*i++) = res2;
      *(data[0] + strides[0]*i++) = res3;
      *(data[0] + strides[0]*i++) = res4;
      *(data[0] + strides[0]*i++) = res5;
      *(data[0] + strides[0]*i++) = res6;
      *(data[0] + strides[0]*i++) = res7;
      *(data[0] + strides[0]*i++) = res8;
      *(data[0] + strides[0]*i++) = res9;
      *(data[0] + strides[0]*i++) = res10;
      *(data[0] + strides[0]*i++) = res11;
      
    }
    if (start + 11 < end) {
      i -= 11;
    }
    while (i < end) {
      scalar_t x = *(data[1] + i * strides[1]);
      scalar_t y = *(data[2] + i * strides[2]);
      scalar_t res = x + y;
      *(data[0] + strides[0]*i++) = res;
    }
  }
  else {
    size_t i = start;
    while ( i + 10 < end){
      scalar_t x1 = *(data[1] + offset(i, input));
      scalar_t y1 = *(data[2] + offset(i, other));
      scalar_t x2 = *(data[1] + offset(i+1, input));
      scalar_t y2 = *(data[2] + offset(i+1, other));
      scalar_t x3 = *(data[1] + offset(i+2, input));
      scalar_t y3 = *(data[2] + offset(i+2, other));
      scalar_t x4 = *(data[1] + offset(i+3, input));
      scalar_t y4 = *(data[2] + offset(i+3, other));
      scalar_t x5 = *(data[1] + offset(i+4, input));
      scalar_t y5 = *(data[2] + offset(i+4, other));
      scalar_t x6 = *(data[1] + offset(i+5, input));
      scalar_t y6 = *(data[2] + offset(i+5, other));
      scalar_t x7 = *(data[1] + offset(i+6, input));
      scalar_t y7 = *(data[2] + offset(i+6, other));
      scalar_t x8 = *(data[1] + offset(i+7, input));
      scalar_t y8 = *(data[2] + offset(i+7, other));
      scalar_t x9 = *(data[1] + offset(i+8, input));
      scalar_t y9 = *(data[2] + offset(i+8, other));
      scalar_t x10 = *(data[1] + offset(i+9, input));
      scalar_t y10 = *(data[2] + offset(i+9, other));
      scalar_t x11 = *(data[1] + offset(i+10, input));
      scalar_t y11 = *(data[2] + offset(i+10, other));

      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      scalar_t res4 = functor(x4, y4);
      scalar_t res5 = functor(x5, y5);
      scalar_t res6 = functor(x6, y6);
      scalar_t res7 = functor(x7, y7);
      scalar_t res8 = functor(x8, y8);
      scalar_t res9 = functor(x9, y9);
      scalar_t res10 = functor(x10, y10);
      scalar_t res11 = functor(x11, y11);

      *(data[0] + offset(i, result)) = res1;
      *(data[0] + offset(i+1, result)) = res2;
      *(data[0] + offset(i+2, result)) = res3;
      *(data[0] + offset(i+3, result)) = res4;
      *(data[0] + offset(i+4, result)) = res5;
      *(data[0] + offset(i+5, result)) = res6;
      *(data[0] + offset(i+6, result)) = res7;
      *(data[0] + offset(i+7, result)) = res8;
      *(data[0] + offset(i+8, result)) = res9;
      *(data[0] + offset(i+9, result)) = res10;
      *(data[0] + offset(i+10, result)) = res11;
      i += 11;
    }
    if (start + 11 < end) {
      i -= 11;
    }
    while (i < end) {
      scalar_t x = *(data[1] + offset(i, input));
      scalar_t y = *(data[2] + offset(i, other));
      scalar_t res = functor(x, y);
      *(data[0] + offset(i, result)) = res;
      i++;
    }
  } 
}


template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll9_try(HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
                               F functor) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t len_per_tile = result.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > result.numel())  ? result.numel() : end;

  // is_trivial_1d
  if(result.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (result.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];

    size_t i = start;
    while ( i + 8 < end){
      scalar_t x1 = *(data[1] + i * strides[1]);
      scalar_t y1 = *(data[2] + i * strides[2]);
      scalar_t x2 = *(data[1] + (i+1) * strides[1]);
      scalar_t y2 = *(data[2] + (i+1) * strides[2]);
      scalar_t x3 = *(data[1] + (i+2) * strides[1]);
      scalar_t y3 = *(data[2] + (i+2) * strides[2]);
      scalar_t x4 = *(data[1] + (i+3) * strides[1]);
      scalar_t y4 = *(data[2] + (i+3) * strides[2]);
      scalar_t x5 = *(data[1] + (i+4) * strides[1]);
      scalar_t y5 = *(data[2] + (i+4) * strides[2]);
      scalar_t x6 = *(data[1] + (i+5) * strides[1]);
      scalar_t y6 = *(data[2] + (i+5) * strides[2]);
      scalar_t x7 = *(data[1] + (i+6) * strides[1]);
      scalar_t y7 = *(data[2] + (i+6) * strides[2]);
      scalar_t x8 = *(data[1] + (i+7) * strides[1]);
      scalar_t y8 = *(data[2] + (i+7) * strides[2]);
      scalar_t x9 = *(data[1] + (i+8) * strides[1]);
      scalar_t y9 = *(data[2] + (i+8) * strides[2]);

      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      scalar_t res4 = functor(x4, y4);
      scalar_t res5 = functor(x5, y5);
      scalar_t res6 = functor(x6, y6);
      scalar_t res7 = functor(x7, y7);
      scalar_t res8 = functor(x8, y8);
      scalar_t res9 = functor(x9, y9);

      *(data[0] + strides[0]*i++) = res1;
      *(data[0] + strides[0]*i++) = res2;
      *(data[0] + strides[0]*i++) = res3;
      *(data[0] + strides[0]*i++) = res4;
      *(data[0] + strides[0]*i++) = res5;
      *(data[0] + strides[0]*i++) = res6;
      *(data[0] + strides[0]*i++) = res7;
      *(data[0] + strides[0]*i++) = res8;
      *(data[0] + strides[0]*i++) = res9;
      
    }
    if (start + 9 < end) {
      i -= 9;
    }
    while (i < end) {
      scalar_t x = *(data[1] + i * strides[1]);
      scalar_t y = *(data[2] + i * strides[2]);
      scalar_t res = x + y;
      *(data[0] + strides[0]*i++) = res;
    }
  }
  else {
    size_t i = start;
    while ( i + 8 < end){
      scalar_t x1 = *(data[1] + offset(i, input));
      scalar_t y1 = *(data[2] + offset(i, other));
      scalar_t x2 = *(data[1] + offset(i+1, input));
      scalar_t y2 = *(data[2] + offset(i+1, other));
      scalar_t x3 = *(data[1] + offset(i+2, input));
      scalar_t y3 = *(data[2] + offset(i+2, other));
      scalar_t x4 = *(data[1] + offset(i+3, input));
      scalar_t y4 = *(data[2] + offset(i+3, other));
      scalar_t x5 = *(data[1] + offset(i+4, input));
      scalar_t y5 = *(data[2] + offset(i+4, other));
      scalar_t x6 = *(data[1] + offset(i+5, input));
      scalar_t y6 = *(data[2] + offset(i+5, other));
      scalar_t x7 = *(data[1] + offset(i+6, input));
      scalar_t y7 = *(data[2] + offset(i+6, other));
      scalar_t x8 = *(data[1] + offset(i+7, input));
      scalar_t y8 = *(data[2] + offset(i+7, other));
      scalar_t x9 = *(data[1] + offset(i+8, input));
      scalar_t y9 = *(data[2] + offset(i+8, other));

      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      scalar_t res4 = functor(x4, y4);
      scalar_t res5 = functor(x5, y5);
      scalar_t res6 = functor(x6, y6);
      scalar_t res7 = functor(x7, y7);
      scalar_t res8 = functor(x8, y8);
      scalar_t res9 = functor(x9, y9);

      *(data[0] + offset(i, result)) = res1;
      *(data[0] + offset(i+1, result)) = res2;
      *(data[0] + offset(i+2, result)) = res3;
      *(data[0] + offset(i+3, result)) = res4;
      *(data[0] + offset(i+4, result)) = res5;
      *(data[0] + offset(i+5, result)) = res6;
      *(data[0] + offset(i+6, result)) = res7;
      *(data[0] + offset(i+7, result)) = res8;
      *(data[0] + offset(i+8, result)) = res9;
      i += 9;
    }
    if (start + 9 < end) {
      i -= 9;
    }
    while (i < end) {
      scalar_t x = *(data[1] + offset(i, input));
      scalar_t y = *(data[2] + offset(i, other));
      scalar_t res = functor(x, y);
      *(data[0] + offset(i, result)) = res;
      i++;
    }
  } 
}

template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll7_try(HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
                               F functor) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t len_per_tile = result.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > result.numel())  ? result.numel() : end;


  // is_trivial_1d
  if(result.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (result.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];

    size_t i = start;
    while ( i + 6 < end){
      scalar_t x1 = *(data[1] + i * strides[1]);
      scalar_t y1 = *(data[2] + i * strides[2]);
      scalar_t x2 = *(data[1] + (i+1) * strides[1]);
      scalar_t y2 = *(data[2] + (i+1) * strides[2]);
      scalar_t x3 = *(data[1] + (i+2) * strides[1]);
      scalar_t y3 = *(data[2] + (i+2) * strides[2]);
      scalar_t x4 = *(data[1] + (i+3) * strides[1]);
      scalar_t y4 = *(data[2] + (i+3) * strides[2]);
      scalar_t x5 = *(data[1] + (i+4) * strides[1]);
      scalar_t y5 = *(data[2] + (i+4) * strides[2]);
      scalar_t x6 = *(data[1] + (i+5) * strides[1]);
      scalar_t y6 = *(data[2] + (i+5) * strides[2]);
      scalar_t x7 = *(data[1] + (i+6) * strides[1]);
      scalar_t y7 = *(data[2] + (i+6) * strides[2]);

      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      scalar_t res4 = functor(x4, y4);
      scalar_t res5 = functor(x5, y5);
      scalar_t res6 = functor(x6, y6);
      scalar_t res7 = functor(x7, y7);

      *(data[0] + strides[0]*i++) = res1;
      *(data[0] + strides[0]*i++) = res2;
      *(data[0] + strides[0]*i++) = res3;
      *(data[0] + strides[0]*i++) = res4;
      *(data[0] + strides[0]*i++) = res5;
      *(data[0] + strides[0]*i++) = res6;
      *(data[0] + strides[0]*i++) = res7;
      
    }
    if (start + 7 < end) {
      i -= 7;
    }
    while (i < end) {
      scalar_t x = *(data[1] + i * strides[1]);
      scalar_t y = *(data[2] + i * strides[2]);
      scalar_t res = x + y;
      *(data[0] + strides[0]*i++) = res;
    }
  } 
  else {
    size_t i = start;
    while ( i + 6 < end){
      scalar_t x1 = *(data[1] + offset(i, input));
      scalar_t y1 = *(data[2] + offset(i, other));
      scalar_t x2 = *(data[1] + offset(i+1, input));
      scalar_t y2 = *(data[2] + offset(i+1, other));
      scalar_t x3 = *(data[1] + offset(i+2, input));
      scalar_t y3 = *(data[2] + offset(i+2, other));
      scalar_t x4 = *(data[1] + offset(i+3, input));
      scalar_t y4 = *(data[2] + offset(i+3, other));
      scalar_t x5 = *(data[1] + offset(i+4, input));
      scalar_t y5 = *(data[2] + offset(i+4, other));
      scalar_t x6 = *(data[1] + offset(i+5, input));
      scalar_t y6 = *(data[2] + offset(i+5, other));
      scalar_t x7 = *(data[1] + offset(i+6, input));
      scalar_t y7 = *(data[2] + offset(i+6, other));

      /*
      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      scalar_t res4 = functor(x4, y4);
      scalar_t res5 = functor(x5, y5);
      scalar_t res6 = functor(x6, y6);
      scalar_t res7 = functor(x7, y7);
      */

      *(data[0] + offset(i, result)) = functor(x1, y1); //res1;
      *(data[0] + offset(i+1, other)) = functor(x2, y2); //res2;
      *(data[0] + offset(i+2, other)) = functor(x3, y3); //res3;
      *(data[0] + offset(i+3, other)) = functor(x4, y4); //res4;
      *(data[0] + offset(i+4, other)) = functor(x5, y5); //res5;
      *(data[0] + offset(i+5, other)) = functor(x6, y6); //res6;
      *(data[0] + offset(i+6, other)) = functor(x7, y7); //res7;
      i += 7;
    }
    if (start + 7 < end) {
      i -= 7;
    }
    while (i < end) {
      scalar_t x = *(data[1] + offset(i, input));
      scalar_t y = *(data[2] + offset(i, other));
      scalar_t res = functor(x, y);
      *(data[0] + offset(i, result)) = res;
      i++;
    }
  }
}

template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll5_try(HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
                               F functor) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t len_per_tile = result.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > result.numel())  ? result.numel() : end;

  // is_trivial_1d
  if(result.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (result.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];

    size_t i = start;
    while ( i + 4 < end){
      scalar_t x1 = *(data[1] + i * strides[1]);
      scalar_t y1 = *(data[2] + i * strides[2]);
      scalar_t x2 = *(data[1] + (i+1) * strides[1]);
      scalar_t y2 = *(data[2] + (i+1) * strides[2]);
      scalar_t x3 = *(data[1] + (i+2) * strides[1]);
      scalar_t y3 = *(data[2] + (i+2) * strides[2]);
      scalar_t x4 = *(data[1] + (i+3) * strides[1]);
      scalar_t y4 = *(data[2] + (i+3) * strides[2]);
      scalar_t x5 = *(data[1] + (i+4) * strides[1]);
      scalar_t y5 = *(data[2] + (i+4) * strides[2]);

      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      scalar_t res4 = functor(x4, y4);
      scalar_t res5 = functor(x5, y5);

      *(data[0] + strides[0]*i++) = res1;
      *(data[0] + strides[0]*i++) = res2;
      *(data[0] + strides[0]*i++) = res3;
      *(data[0] + strides[0]*i++) = res4;
      *(data[0] + strides[0]*i++) = res5;
    }
    if (start + 5 < end) {
      i -= 5;
    }
    while (i < end) {
      scalar_t x = *(data[1] + i * strides[1]);
      scalar_t y = *(data[2] + i * strides[2]);
      scalar_t res = x + y;
      *(data[0] + strides[0]*i++) = res;
    }
  } 
  else {
    size_t i = start;
    while ( i + 4 < end){
      scalar_t x1 = *(data[1] + offset(i, input));
      scalar_t y1 = *(data[2] + offset(i, other));
      scalar_t x2 = *(data[1] + offset(i+1, input));
      scalar_t y2 = *(data[2] + offset(i+1, other));
      scalar_t x3 = *(data[1] + offset(i+2, input));
      scalar_t y3 = *(data[2] + offset(i+2, other));
      scalar_t x4 = *(data[1] + offset(i+3, input));
      scalar_t y4 = *(data[2] + offset(i+3, other));
      scalar_t x5 = *(data[1] + offset(i+4, input));
      scalar_t y5 = *(data[2] + offset(i+4, other));

      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      scalar_t res4 = functor(x4, y4);
      scalar_t res5 = functor(x5, y5);

      *(data[0] + offset(i, result)) = res1;
      *(data[0] + offset(i+1, other)) = res2;
      *(data[0] + offset(i+2, other)) = res3;
      *(data[0] + offset(i+3, other)) = res4;
      *(data[0] + offset(i+4, other)) = res5;
      i += 5;
    }
    if (start + 5 < end) {
      i -= 5;
    }
    while (i < end) {
      scalar_t x = *(data[1] + offset(i, input));
      scalar_t y = *(data[2] + offset(i, other));
      scalar_t res = functor(x, y);
      *(data[0] + offset(i, result)) = res;
      i++;
    }
  }
}

template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll3_try(HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
                               F functor) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t len_per_tile = result.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > result.numel())  ? result.numel() : end;

  // is_trivial_1d
  if(result.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (result.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];
    
    //data[0] += strides[0] * start;
    //data[1] += strides[1] * start;
    //data[2] += strides[2] * start;
    

    size_t i = start;
    while ( i + 2 < end){
      /*
      scalar_t x1 = data[1][(i) * strides[1]];   //  *(data[1] + i * strides[1]);
      scalar_t y1 = data[2][(i) * strides[2]];   //  *(data[2] + i * strides[2]);
      scalar_t x2 = data[1][(i+1) * strides[1]]; //  *(data[1] + (i+1) * strides[1]);
      scalar_t y2 = data[2][(i+1) * strides[2]]; //  *(data[2] + (i+1) * strides[2]);
      scalar_t x3 = data[1][(i+2) * strides[1]]; //  *(data[1] + (i+2) * strides[1]);
      scalar_t y3 = data[2][(i+2) * strides[2]]; //  *(data[2] + (i+2) * strides[2]);
      */
      scalar_t x1 = *(data[1] + i * strides[1]);
      scalar_t y1 = *(data[2] + i * strides[2]);
      scalar_t x2 = *(data[1] + (i+1) * strides[1]);
      scalar_t y2 = *(data[2] + (i+1) * strides[2]);
      scalar_t x3 = *(data[1] + (i+2) * strides[1]);
      scalar_t y3 = *(data[2] + (i+2) * strides[2]);

      /*
      scalar_t x1 = *data[1];
      scalar_t y1 = *data[2];
      data[1] += strides[1];
      data[2] += strides[2];
      scalar_t x2 = *data[1];
      scalar_t y2 = *data[2];
      data[1] += strides[1];
      data[2] += strides[2];
      scalar_t x3 = *data[1];
      scalar_t y3 = *data[2];
      data[1] += strides[1];
      data[2] += strides[2];

      *data[0] = functor(x1, y1);
      data[0] += strides[0];
      *data[0] = functor(x2, y2);
      data[0] += strides[0];
      *data[0] = functor(x3, y3);
      data[0] += strides[0];
      
      i += 3;
      */
      
      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      
      *(data[0] + strides[0] * i++) = res1;
      *(data[0] + strides[0] * i++) = res2;
      *(data[0] + strides[0] * i++) = res3;

      /*
      data[0][strides[0] * i++] = functor(x1, y1);
      data[0][strides[0] * i++] = functor(x2, y2);
      data[0][strides[0] * i++] = functor(x3, y3);
      */
    }
    if (start + 3 < end) {
      i -= 3;
    }
    while (i < end) {
      scalar_t x = *(data[1] + i * strides[1]);
      scalar_t y = *(data[2] + i * strides[2]);
      scalar_t res = functor(x, y);
      *(data[0] + strides[0]*i++) = res;
    }
    
  }
  else {
    size_t i = start;
    while ( i + 2 < end){
      scalar_t x1 = *(data[1] + offset(i, input));
      scalar_t y1 = *(data[2] + offset(i, other));
      scalar_t x2 = *(data[1] + offset(i+1, input));
      scalar_t y2 = *(data[2] + offset(i+1, other));
      scalar_t x3 = *(data[1] + offset(i+2, input));
      scalar_t y3 = *(data[2] + offset(i+2, other));
      
      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      
      *(data[0] + offset(i, result)) = res1; 
      *(data[0] + offset(i+1, input)) = res2;
      *(data[0] + offset(i+2, input)) = res3;
      i += 3;
    }
    if (start + 3 < end) {
      i -= 3;
    }
    while (i < end) {
      scalar_t x = *(data[1] + offset(i, input));
      scalar_t y = *(data[2] + offset(i, other));
      scalar_t res = functor(x, y);
      *(data[0] + offset(i, result)) = res;
      i++;
    }
  } 
}

template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll1_try(HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
                               F functor) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t len_per_tile = result.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > result.numel())  ? result.numel() : end;

  // is_trivial_1d
  if(result.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (result.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];

    size_t i = start;
    while (i < end) {
      scalar_t x = *(data[1] + i * strides[1]);
      scalar_t y = *(data[2] + i * strides[2]);
      scalar_t res = functor(x, y);
      *(data[0] + strides[0]*i++) = res;
    }
  }
  else {
    size_t i = start;
    while (i < end) {
      scalar_t x = *(data[1] + offset(i, input));
      scalar_t y = *(data[2] + offset(i, other));
      scalar_t res = functor(x, y);
      *(data[0] + offset(i, result)) = res;
      i++;
    }
  } 
}

/*
template<int N, typename scalar_t, typename F>
struct Unroll {
  inline static void copy_from(scalar_t* src, scalar_t* dest, size_t i);
  inline static void copy_to(scalar_t* src, scalar_t* dest, size_t i);
  inline static void compute(scalar_t* res, scalar_t* x, scalar_t* y, F functor);
};

template<int N, typename scalar_t, typename F>
inline void Unroll<N, scalar_t, F>::copy_from(scalar_t* src, register scalar_t* dest, size_t i){
  dest[N] = src[i + N];
  //asm volatile("": : :"memory");

  Unroll<N-1, scalar_t, F>::copy_from(src, dest, i);
}

template<int N, typename scalar_t, typename F>
inline void Unroll<N, scalar_t, F>::copy_to(scalar_t* src, scalar_t* dest, size_t i){
  dest[N + i] = src[N];
  //asm volatile("mfen": : :"memory");

  Unroll<N-1, scalar_t, F>::copy_to(src, dest, i);
}

template<int N, typename scalar_t, typename F>
inline void Unroll<N, scalar_t, F>::compute(scalar_t* res, scalar_t* x, scalar_t* y, F functor){
  res[N] = functor(x[N], y[N]);
  //asm volatile("mfen": : :"memory");

  Unroll<N-1, scalar_t, F>::compute(res, x, y, functor);
}

template<typename scalar_t, typename F>
struct Unroll<0, scalar_t, F> {
  inline static void copy_from(scalar_t* src, scalar_t* dest, size_t i);
  inline static void copy_to(scalar_t* src, scalar_t* dest, size_t i);
  inline static void compute(scalar_t* res, scalar_t* x, scalar_t* y, F functor);
};

template<typename scalar_t, typename F>
inline void Unroll<0, scalar_t, F>::copy_from(scalar_t* src, scalar_t* dest, size_t i){
  dest[0] = src[i];
}

template<typename scalar_t, typename F>
inline void Unroll<0, scalar_t, F>::copy_to(scalar_t* src, scalar_t* dest, size_t i){
  dest[i] = src[0];
}

template<typename scalar_t, typename F>
inline void Unroll<0, scalar_t, F>::compute(scalar_t* res, scalar_t* x, scalar_t* y, F functor){
  res[0] = functor(x[0], y[0]);
}


template< int N, typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll_pragma(HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
                               F functor) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t len_per_tile = result.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > result.numel())  ? result.numel() : end;

  // is_trivial_1d
  if(result.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (result.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];

    register scalar_t x[N];
    register scalar_t y[N];
    register scalar_t res[N];

    size_t i = start;
    while (i + N < end) {
      Unroll<N-1, scalar_t, F>::copy_from(data[1], x, i);
      Unroll<N-1, scalar_t, F>::copy_from(data[2], y, i);

      //asm volatile("": : :"memory");
      
      Unroll<N-1, scalar_t, F>::compute(res, x, y, functor);
      
      Unroll<N-1, scalar_t, F>::copy_to(res, data[0], i);
      
      i += N;
    }
    if (start + N < end) {
      i -= N;
    }
    while (i < end) {
      scalar_t x = *(data[1] + i * strides[1]);
      scalar_t y = *(data[2] + i * strides[2]);
      scalar_t res = functor(x, y);
      *(data[0] + strides[0]*i++) = res;
    }
  }
  
  else {
    size_t i = start;
    while (i < end) {
      scalar_t x = *(data[1] + offset(i, input));
      scalar_t y = *(data[2] + offset(i, other));
      scalar_t res = functor(x, y);
      *(data[0] + offset(i, result)) = res;
      i++;
    }
  } 
  
}
*/

template<typename scalar_t, typename F>
void hb_tiled_foreach_unroll_macro(HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
                               F functor) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t len_per_tile = result.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > result.numel())  ? result.numel() : end;

  // is_trivial_1d
  if(result.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (result.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];

    for (size_t i = start; i < end; i++) {
      scalar_t x = *(data[1] + i * strides[1]);
      scalar_t y = *(data[2] + i * strides[2]);
      scalar_t res = functor(x, y);
      *(data[0] + strides[0]*i++) = res;
    }
  }
}



extern "C" {

  __attribute__ ((noinline))  int tensorlib_vvadd(
          hb_tensor_t* result_p,
          hb_tensor_t* self_p,
          hb_tensor_t* other_p) {

    // Tutorial TODO:
    // Convert all low level pointers to Tensor objects
    HBTensor<float> result(result_p);
    HBTensor<float> self(self_p);
    HBTensor<float> other(other_p);

    
    float* res_ptr = (float*)result.data_ptr();
    float* self_ptr = (float*)self.data_ptr();
    float* other_ptr = (float*)other.data_ptr();

    size_t size = result.numel();
    

    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    
    /*
    hb_tiled_foreach_unroll7(result, self, other, 
      [&](float self, float other) {
        return self + other;
      }
    );
    */
    /*
    hb_tiled_foreach_unroll_macro(result, self, other, 
      [&](float self, float other) {
        return self + other;
      }
    );
    */
    
    hb_tiled_foreach_unroll<4>(result, self, other, 
      [&](float self, float other) {
        return self + other;
      }
    );
    
    /*
    if (__bsg_id == 0) {
      size_t st = result.ndim() - 1;

      uint32_t strides[3];
      strides[0] = (result.get_strides())[st];
      strides[1] = (self.get_strides())[st];
      strides[2] = (other.get_strides())[st];

      for (size_t i = 0; i < result.numel(); i++) {
        res_ptr[i * strides[0]] = self_ptr[i * strides[1]] + other_ptr[i * strides[2]];
      }

    }
    */
    /*
    for (size_t i = 0; i < self.ndim(); i++) {
      res_ptr[i] = self.get_strides()[i]; //self_ptr[i];
    }
    */
    /*
    // Parallel add
    size_t len_per_tile = size / (bsg_tiles_X * bsg_tiles_Y) + 1;

    size_t start = __bsg_id * len_per_tile;
    size_t end = start + len_per_tile;
    end = end > size ? size : end;
    */

    //if (start < size) {
      /*
      for (size_t i = start; i < end; i++) {
        //res_ptr[i * result_dim * result_st] = other_ptr[i * other_dim * other_st];
        //result(i) = self(i); 

        
        result(i) = self(i) + other(i);
        
        float x1 = self(i);
        float y1 = other(i);
        float x2, y2;
        if (i+1 < end) {
          x2 = self(i+1);
          y2 = other(i+1);
        }
        result(i) = x1 + y1;
        if (i+1 < end) {
          result(i+1) = x2 + y2;
        }
        
      }
      */
      
      /*
      size_t i, j;
      for (i = start, j = start+1; i < end && j < end; i++, j++) {
        float x1 = self(i);
        float y1 = other(i);
        float x2 = self(j);
        float y2 = other(j);
        //result(i) = self(i) + other(i);
        result(i) = x1 + y1;
        result(j) = x2 + y2;
      }
      if (i < end) {
        float x1 = self(i);
        float y1 = other(i);
        result(i) = x1 + y1;
      }
      */

      /*
      float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
      float y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15;

      size_t i = start;
      //size_t j = start+1;
      //size_t k = start+2;
      while( i + 14 < end ){ // && j < end && k < end) {
        x1 = self_ptr[i];
        y1 = other_ptr[i];
        x2 = self_ptr[(i+1)];
        y2 = other_ptr[(i+1)];
        x3 = self_ptr[(i+2)];
        y3 = other_ptr[(i+2)];
        x4 = self_ptr[(i+3)];
        y4 = other_ptr[(i+3)];
        x5 = self_ptr[(i+4)];
        y5 = other_ptr[(i+4)];
        x6 = self_ptr[(i+5)];
        y6 = other_ptr[(i+5)];
        x7 = self_ptr[(i+6)];
        y7 = other_ptr[(i+6)];
        x8 = self_ptr[(i+7)];
        y8 = other_ptr[(i+7)];
        x9 = self_ptr[(i+8)];
        y9 = other_ptr[(i+8)];
        x10 = self_ptr[(i+9)];
        y10 = other_ptr[(i+9)];
        x11 = self_ptr[(i+10)];
        y11 = other_ptr[(i+10)];
        x12 = self_ptr[(i+11)];
        y12 = other_ptr[(i+11)];
        x13 = self_ptr[(i+12)];
        y13 = other_ptr[(i+12)];
        x14 = self_ptr[(i+13)];
        y14 = other_ptr[(i+13)];
        x15 = self_ptr[(i+14)];
        y15 = other_ptr[(i+14)];
        
        res_ptr[i++] = x1 + y1;
        res_ptr[i++] = x2 + y2;
        res_ptr[i++] = x3 + y3;
        res_ptr[i++] = x4 + y4;
        res_ptr[i++] = x5 + y5;
        res_ptr[i++] = x6 + y6;
        res_ptr[i++] = x7 + y7;
        res_ptr[i++] = x8 + y8;
        res_ptr[i++] = x9 + y9;
        res_ptr[i++] = x10 + y10;
        res_ptr[i++] = x11 + y11;
        res_ptr[i++] = x12 + y12;
        res_ptr[i++] = x13 + y13;
        res_ptr[i++] = x14 + y14;
        res_ptr[i++] = x15 + y15;
        
        //i += 10;
      }
      
      if (start + 15 < end) {
        i -= 15;
      }
      while (i < end) {
        float x = self_ptr[i];
        float y = other_ptr[i];
        float res = x + y;
        res_ptr[i++] = res;
      }
      */      

      /* 
      float x[10];
      float y[10];
      float res[10];

      size_t i = start;
      while ( i + 10 < end ) {
        
        memcpy(x, self_ptr + i, 10 * sizeof(float));
        memcpy(y, other_ptr + i, 10 * sizeof(float));
        

        for (size_t j = 0; j < 10; j++) {
          x[j] = self_ptr[i + j];
          y[j] = self_ptr[i + j];
        }

        for ( size_t j = 0; j < 10; j++ ) {
          res[j] = x[j] + y[j];
        }

        for (size_t j = 0; j < 10; j++) {
          res_ptr[i++] = res[j];
        }

        //memcpy(res_ptr + i, res, 10 * sizeof(float));
        
        //i += 10;
      }
      if (start + 10 < end) {
        i -= 10;
      }
      while ( i < end ) {
        res_ptr[i] = self_ptr[i] + other_ptr[i];
        i++; 
      }
      */

      /*
      float x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;
      float y1, y2, y3, y4, y5, y6, y7, y8, y9, y10;

      size_t i = start;
      //size_t j = start+1;
      //size_t k = start+2;
      while( i + 10 < end ){ // && j < end && k < end) {
        x1 = self(i);
        y1 = other(i);
        x2 = self(i+1);
        y2 = other(i+1);
        x3 = self(i+2);
        y3 = other(i+2);
        x4 = self(i+3);
        y4 = other(i+3);
        x5 = self(i+4);
        y5 = other(i+4);
        x6 = self(i+5);
        y6 = other(i+5);
        x7 = self(i+6);
        y7 = other(i+6);
        x8 = self(i+7);
        y8 = other(i+7);
        x9 = self(i+8);
        y9 = other(i+8);
        x10 = self(i+9);
        y10 = other(i+9);

        result(i++) = x1 + y1;
        result(i++) = x2 + y2;
        result(i++) = x3 + y3;
        result(i++) = x4 + y4;
        result(i++) = x5 + y5;
        result(i++) = x6 + y6;
        result(i++) = x7 + y7;
        result(i++) = x8 + y8;
        result(i++) = x9 + y9;
        result(i++) = x10 + y10;
  
        //i += 10;
      }
      if (start + 10 < end) {
        i -= 10;
      }
      while (i < end) {
        float x = self(i);
        float y = other(i);
        float res = x + y;
        result(i++) = res;
      }
      */
    //}

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_vvadd, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
