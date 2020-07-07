//====================================================================
// Element-wise for helper function
// 03/12/2020 Lin Cheng (lc873@cornell.edu)
//
// Note: assuming a 3D tensor, and you access it with (x, y, z)
// Plain tensor has indices numbered as (0, 1, 2)
// BUT iterator tensor has indices numbered as (2, 1, 0)
//====================================================================

#ifndef _HB_TILED_FOR_HPP
#define _HB_TILED_FOR_HPP

#include <map>
#include <math.h>
#include <initializer_list>
#include <hb_assert.hpp>
#include <hb_tensor.hpp>

// =========================================================
// Linear index to offset
// =========================================================
template<typename scalar_t>
inline uint32_t offset_calc(uint32_t idx, HBTensor<scalar_t> tensor) {
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

// =========================================================
// Tiled range calculation
// hb_range -> [start, end)
// =========================================================
typedef struct hb_range {
  size_t start;
  size_t end;
} hb_range;

inline void calc_range(hb_range* range, size_t numel) {
  // per pod chunk
  size_t len_per_pod  = numel / BSG_POD_DIM + 1;
  // chunk range
  size_t pod_start    = len_per_pod * __bsg_pod_id;
  size_t pod_end      = pod_start + len_per_pod;
  pod_end = (pod_end > numel) ? numel : pod_end;
  if (pod_start >= pod_end) {
    range->start = 0;
    range->end   = 0;
    return;
  }
  size_t pod_size     = pod_end - pod_start;

  // per tile range within a pod
  size_t len_per_tile = pod_size / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start        = len_per_tile * __bsg_id;
  size_t end          = start + len_per_tile;
  end = (end > pod_size) ? pod_size : end;
  if (start >= end) {
    range->start = 0;
    range->end   = 0;
    return;
  }

  // range in global idx
  range->start = pod_start + start;
  range->end   = pod_start + end;

  return;
}

// =========================================================
// Tiled Pointwise for
// =========================================================

template<typename scalar_t, typename F, class... Types>
inline void hb_tiled_foreach(F functor,
                             HBTensor<scalar_t> res,
                             Types... args) {
  // Iterating over all elementes
  hb_range range;
  calc_range(&range, res.numel());
  size_t start = range.start;
  size_t end   = range.end;

  // Static dispatch based on number number of operands
  hb_tiled_foreach_impl(
      start, end, functor, res,
      args...,
      (__remote scalar_t*) res.data_ptr(),
      ((__remote scalar_t*) args.data_ptr())...);
}

// Nullary
template<typename scalar_t, typename F, typename... P>
__attribute__((noinline)) void hb_tiled_foreach_impl(
      size_t start, size_t end, F functor,
      HBTensor<scalar_t> res,
      __remote scalar_t* NOALIAS res_ptr) {
  // is_trivial_1d
  if(res.ndim() == 1) {
    UNROLL(16) for(size_t idx = start; idx < end; idx++) {
      res_ptr[idx * res.get_strides()[0]] =
        functor();
    }
  } else {
    UNROLL(16) for (size_t idx = start; idx < end; idx++) {
      res_ptr[offset_calc(idx, res)] =
        functor();
    }
  }
}

// Unary
template<typename scalar_t, typename F, typename... P>
__attribute__((noinline)) void hb_tiled_foreach_impl(
      size_t start, size_t end, F functor,
      HBTensor<scalar_t> res,
      HBTensor<scalar_t> tensor_arg0,
      __remote scalar_t* NOALIAS res_ptr,
      __remote scalar_t* NOALIAS tensor_data_ptr0) {
  // is_trivial_1d
  if(res.ndim() == 1) {
    UNROLL(16) for(size_t idx = start; idx < end; idx++) {
      res_ptr[idx * res.get_strides()[0]] =
        functor(tensor_data_ptr0[idx * tensor_arg0.get_strides()[0]]);
    }
  } else {
    UNROLL(16) for (size_t idx = start; idx < end; idx++) {
      res_ptr[offset_calc(idx, res)] =
        functor(tensor_data_ptr0[offset_calc(idx, tensor_arg0)]);
    }
  }
}

// Binary
template<typename scalar_t, typename F, typename... P>
__attribute__((noinline)) void hb_tiled_foreach_impl(
      size_t start, size_t end, F functor,
      HBTensor<scalar_t> res,
      HBTensor<scalar_t> tensor_arg0,
      HBTensor<scalar_t> tensor_arg1,
      __remote scalar_t* NOALIAS res_ptr,
      __remote scalar_t* NOALIAS tensor_data_ptr0,
      __remote scalar_t* NOALIAS tensor_data_ptr1) {
  // is_trivial_1d
  if(res.ndim() == 1) {
    UNROLL(16) for(size_t idx = start; idx < end; idx++) {
      res_ptr[idx * res.get_strides()[0]] =
        functor(tensor_data_ptr0[idx * tensor_arg0.get_strides()[0]],
                tensor_data_ptr1[idx * tensor_arg1.get_strides()[0]]);
    }
  } else {
    UNROLL(16) for (size_t idx = start; idx < end; idx++) {
      res_ptr[offset_calc(idx, res)] =
        functor(tensor_data_ptr0[offset_calc(idx, tensor_arg0)],
                tensor_data_ptr1[offset_calc(idx, tensor_arg1)]);
    }
  }
}

// Ternary
template<typename scalar_t, typename F, typename... P>
__attribute__((noinline)) void hb_tiled_foreach_impl(
      size_t start, size_t end, F functor,
      HBTensor<scalar_t> res,
      HBTensor<scalar_t> tensor_arg0,
      HBTensor<scalar_t> tensor_arg1,
      HBTensor<scalar_t> tensor_arg2,
      __remote scalar_t* NOALIAS res_ptr,
      __remote scalar_t* NOALIAS tensor_data_ptr0,
      __remote scalar_t* NOALIAS tensor_data_ptr1,
      __remote scalar_t* NOALIAS tensor_data_ptr2) {
  // is_trivial_1d
  if(res.ndim() == 1) {
    UNROLL(16) for(size_t idx = start; idx < end; idx++) {
      res_ptr[idx * res.get_strides()[0]] =
        functor(tensor_data_ptr0[idx * tensor_arg0.get_strides()[0]],
                tensor_data_ptr1[idx * tensor_arg1.get_strides()[0]],
                tensor_data_ptr2[idx * tensor_arg2.get_strides()[0]]);
    }
  } else {
    UNROLL(16) for (size_t idx = start; idx < end; idx++) {
      res_ptr[offset_calc(idx, res)] =
        functor(tensor_data_ptr0[offset_calc(idx, tensor_arg0)],
                tensor_data_ptr1[offset_calc(idx, tensor_arg1)],
                tensor_data_ptr2[offset_calc(idx, tensor_arg2)]);
    }
  }
}

// =========================================================
// Tile Element-wise for -- Unary ops -- Special conversion
//
// This function calculates the per tile range automatically
//==========================================================

template<typename scalar_src, typename scalar_dst, typename F>
inline void hb_tiled_foreach_conversion(HBTensor<scalar_dst> res,
                               HBTensor<scalar_src> input,
                               F functor) {

  __remote scalar_dst* res_data = (__remote scalar_dst*)res.data_ptr();
  __remote scalar_src* input_data = (__remote scalar_src*)input.data_ptr();

  // is_trivial_1d
  if(res.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[2];
    strides[0] = (res.get_strides())[0];
    strides[1] = (input.get_strides())[0];

    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    hb_range range;
    calc_range(&range, res.numel());
    size_t start = range.start;
    size_t end   = range.end;

    res_data += strides[0] * start;
    input_data += strides[1] * start;
    size_t idx = start;
    if (end - start > 4) {
      for (; idx < end - 4; idx += 4) {
        scalar_src input_dp_0 = *(input_data);
        __remote scalar_dst* res_dp_0 = (res_data);
        res_data += strides[0];
        input_data += strides[1];

        scalar_src input_dp_1 = *(input_data);
        __remote scalar_dst* res_dp_1 = (res_data);
        res_data += strides[0];
        input_data += strides[1];

        scalar_src input_dp_2 = *(input_data);
        __remote scalar_dst* res_dp_2 = (res_data);
        res_data += strides[0];
        input_data += strides[1];

        scalar_src input_dp_3 = *(input_data);
        __remote scalar_dst* res_dp_3 = (res_data);
        res_data += strides[0];
        input_data += strides[1];

        *res_dp_0 = functor(input_dp_0);
        *res_dp_1 = functor(input_dp_1);
        *res_dp_2 = functor(input_dp_2);
        *res_dp_3 = functor(input_dp_3);
      }
    }
    for (; idx < end; idx++) {
      __remote scalar_dst* res_dp = (res_data);
      __remote scalar_src* input_dp = (input_data);
      *res_dp = functor(*input_dp);
      res_data += strides[0];
      input_data += strides[1];
    }
  } else if (res.ndim() == 2) {
    // the idea is each tile takes care of the first dim in one shot
    hb_range range;
    calc_range(&range, res.dim(0));
    size_t start = range.start;
    size_t end   = range.end;

    uint32_t* src_strides = input.get_strides();
    uint32_t* src_sizes = input.get_sizes();
    uint32_t* dst_strides = res.get_strides();
    uint32_t* dst_sizes = res.get_sizes();

    for (size_t idx = start; idx < end; idx++) {
      __remote scalar_dst* dst_data = res_data + idx * dst_strides[0];
      __remote scalar_src* src_data = input_data + idx * src_strides[0];

      for (size_t inner = 0; inner < res.dim(1); inner++) {
        scalar_src input_dp_0 = *(src_data);
        __remote scalar_dst* res_dp_0 = (dst_data);
        dst_data += dst_strides[1];
        src_data += src_strides[1];

        *res_dp_0 = functor(input_dp_0);
      }
    }
  } else if (res.ndim() == 3) {
    hb_range range;
    calc_range(&range, res.dim(0) * res.dim(1));
    size_t start = range.start;
    size_t end   = range.end;

    uint32_t* src_strides = input.get_strides();
    uint32_t* src_sizes = input.get_sizes();
    uint32_t* dst_strides = res.get_strides();
    uint32_t* dst_sizes = res.get_sizes();

    for (size_t idx = start; idx < end; idx++) {
      __remote scalar_dst* dst_data = res_data + idx % dst_sizes[1] * dst_strides[1] + idx / dst_sizes[1] * dst_strides[0];
      __remote scalar_src* src_data = input_data + idx % src_sizes[1] * src_strides[1] + idx / src_sizes[1] * src_strides[0];

      for (size_t inner = 0; inner < res.dim(2); inner++) {
        scalar_src input_dp_0 = *(src_data);
        __remote scalar_dst* res_dp_0 = (dst_data);
        dst_data += dst_strides[2];
        src_data += src_strides[2];

        *res_dp_0 = functor(input_dp_0);
      }
    }
  } else {
    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    hb_range range;
    calc_range(&range, res.numel());
    size_t start = range.start;
    size_t end   = range.end;

    for (size_t idx = start; idx < end; idx++) {
      __remote scalar_dst* res_dp = (res_data + offset_calc(idx, res));
      __remote scalar_src* input_dp = (input_data + offset_calc(idx, input));
      *res_dp = functor(*input_dp);
    }
  }
}

// =========================================================
// HB for
// =========================================================
// functor takes in current index

template <class FetchFunctor>
inline void hb_for(size_t numel, FetchFunctor functor) {
  //--------------------------------------
  // calculate start and end for this tile
  //--------------------------------------
  size_t start = 0;
  size_t end = numel;
  //-----------------
  // loop
  //----------------
  for (size_t i = start; i < end; i++) {
    functor(i);
  }
}

// =========================================================
// HB tile for
// =========================================================
// functor takes in current index

template <class FetchFunctor>
inline void hb_tiled_for(size_t numel, FetchFunctor functor) {
  //--------------------------------------
  // calculate start and end for this tile
  //--------------------------------------
  hb_range range;
  calc_range(&range, numel);
  size_t start = range.start;
  size_t end   = range.end;

  //-----------------
  // loop
  //----------------
  for (size_t i = start; i < end; i++) {
    functor(i);
  }
}

// =========================================================
// HB tile range
// =========================================================
// functor takes in current index

template <class FetchFunctor>
inline void hb_tiled_range(size_t numel, FetchFunctor functor) {
  //--------------------------------------
  // calculate start and end for this tile
  //--------------------------------------
  hb_range range;
  calc_range(&range, numel);
  size_t start = range.start;
  size_t end   = range.end;

  //-----------------
  // range
  //----------------
  functor(start, end);
}



// =========================================================
// Tilewise element op helper - Multidimensional Tensor 
// =========================================================
template<typename scalar_t, typename F>
inline void hb_tiled_foreach_multi(HBTensor<scalar_t> res,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
                               F functor) {
  scalar_t* data[3];
  data[0] = (scalar_t*)res.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  hb_range range;
  calc_range(&range, res.numel());
  size_t start = range.start;
  size_t end   = range.end;

  for (size_t idx = start; idx < end; idx++) {
    scalar_t* res_dp = (data[0] + offset_calc(idx, res));
    scalar_t* input_dp = (data[1] + offset_calc(idx, input));
    scalar_t* other_dp = (data[2] + offset_calc(idx, other));
    *res_dp = functor(*input_dp, *other_dp);
  }

}


// =========================================================
// Tilewise element op helper - Binary 
// Unroll 3
// =========================================================
template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll3(F functor, 
                               HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  hb_range range;
  calc_range(&range, result.numel());
  size_t start = range.start;
  size_t end   = range.end;

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
    while (i + 2 < end){
    
      scalar_t x1 = *(data[1] + i * strides[1]);
      scalar_t y1 = *(data[2] + i * strides[2]);
      scalar_t x2 = *(data[1] + (i+1) * strides[1]);
      scalar_t y2 = *(data[2] + (i+1) * strides[2]);
      scalar_t x3 = *(data[1] + (i+2) * strides[1]);
      scalar_t y3 = *(data[2] + (i+2) * strides[2]);
      
      scalar_t res1 = functor(x1, y1);
      scalar_t res2 = functor(x2, y2);
      scalar_t res3 = functor(x3, y3);
      
      *(data[0] + strides[0] * i++) = res1;
      *(data[0] + strides[0] * i++) = res2;
      *(data[0] + strides[0] * i++) = res3;
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
    hb_tiled_foreach_multi(result, input, other, functor);
  } 
}

// =========================================================
// Tilewise element op helper - Binary 
// Unroll 5
// =========================================================
template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll5(F functor, 
                               HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  hb_range range;
  calc_range(&range, result.numel());
  size_t start = range.start;
  size_t end   = range.end;

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
    hb_tiled_foreach_multi(result, input, other, functor);
  }
}

// =========================================================
// Tilewise element op helper - Binary 
// Unroll 7
// =========================================================
template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll7(F functor, 
                               HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  hb_range range;
  calc_range(&range, result.numel());
  size_t start = range.start;
  size_t end   = range.end;

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
    hb_tiled_foreach_multi(result, input, other, functor);
  }
}

// =========================================================
// Tilewise element op helper - Binary 
// Unroll 9
// =========================================================
template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll9(F functor, 
                               HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  hb_range range;
  calc_range(&range, result.numel());
  size_t start = range.start;
  size_t end   = range.end;

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
    hb_tiled_foreach_multi(result, input, other, functor);
  } 
}

// =========================================================
// Tilewise element op helper - Binary 
// Unroll 11
// =========================================================
template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll11(F functor, 
                               HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  hb_range range;
  calc_range(&range, result.numel());
  size_t start = range.start;
  size_t end   = range.end;

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
    hb_tiled_foreach_multi(result, input, other, functor);
  } 
}

// =========================================================
// Tilewise element op helper - Binary 
// Unroll 13
// =========================================================
template<typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll13(F functor, 
                               HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  hb_range range;
  calc_range(&range, result.numel());
  size_t start = range.start;
  size_t end   = range.end;

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
    hb_tiled_foreach_multi(result, input, other, functor);
  } 
}


#endif
