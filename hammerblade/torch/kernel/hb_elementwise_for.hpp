//====================================================================
// Element-wise for helper function
// 03/12/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#ifndef _BRG_ELEMENT_FOR_HPP
#define _BRG_ELEMENT_FOR_HPP

#include <map>
#include <math.h>
#include <initializer_list>
#include <bsg_assert.hpp>
#include <bsg_tensor.hpp>

// =========================================================
// Linear index to offset
// =========================================================
template<typename scalar_t>
inline uint32_t offset_calc(uint32_t idx, BSGTensor<scalar_t> tensor) {
  uint32_t* strides = tensor.get_strides();
  uint32_t* sizes = tensor.get_sizes();
  uint32_t factor = 1;
  uint32_t offset = 0;
  for(int32_t i = tensor.ndim() - 1; i >= 0; i--) {
    uint32_t dimx = idx % sizes[i];
    offset += dimx * strides[i];
    idx /= factor;
    factor *= sizes[i];
  }
  return offset;
}

// =========================================================
// Elementwise for -- Binary ops
// =========================================================

template<typename scalar_t, typename F>
inline void hb_elementwise_for(BSGTensor<scalar_t> res,
                               BSGTensor<scalar_t> input,
                               BSGTensor<scalar_t> other,
                               F functor) {
  char* data[3];
  data[0] = res.data_ptr();
  data[1] = input.data_ptr();
  data[2] = other.data_ptr();

  // is_trivial_1d
  if(res.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (res.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];

    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t start = 0;
    size_t end = res.numel();
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + strides[0] * idx);
      scalar_t* input_dp = (scalar_t*)(data[1] + strides[1] * idx);
      scalar_t* other_dp = (scalar_t*)(data[2] + strides[2] * idx);
      *res_dp = functor(*input_dp, *other_dp);
    }
  } else {
    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t start = 0;
    size_t end = res.numel();
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + offset_calc(idx, res));
      scalar_t* input_dp = (scalar_t*)(data[1] + offset_calc(idx, input));
      scalar_t* other_dp = (scalar_t*)(data[2] + offset_calc(idx, other));
      *res_dp = functor(*input_dp, *other_dp);
    }
  }
}

// =========================================================
// Elementwise for -- Unary ops
// =========================================================

template<typename scalar_t, typename F>
inline void hb_elementwise_for(BSGTensor<scalar_t> res,
                               BSGTensor<scalar_t> input,
                               F functor) {
  char* data[2];
  data[0] = res.data_ptr();
  data[1] = input.data_ptr();

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
    size_t start = 0;
    size_t end = res.numel();
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + strides[0] * idx);
      scalar_t* input_dp = (scalar_t*)(data[1] + strides[1] * idx);
      *res_dp = functor(*input_dp);
    }
  } else {
    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t start = 0;
    size_t end = res.numel();
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + offset_calc(idx, res));
      scalar_t* input_dp = (scalar_t*)(data[1] + offset_calc(idx, input));
      *res_dp = functor(*input_dp);
    }
  }
}

// =========================================================
// Elementwise for -- Nullary ops
// =========================================================

template<typename scalar_t, typename F>
inline void hb_elementwise_for(BSGTensor<scalar_t> res,
                               F functor) {
  char* data[1];
  data[0] = res.data_ptr();

  // is_trivial_1d
  if(res.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[1];
    strides[0] = (res.get_strides())[0];

    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t start = 0;
    size_t end = res.numel();
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + strides[0] * idx);
      *res_dp = functor();
    }
  } else {
    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t start = 0;
    size_t end = res.numel();
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + offset_calc(idx, res));
      *res_dp = functor();
    }
  }
}

// =========================================================
// Tile Element-wise for -- Binary ops
//
// This function calculates the per tile range automatically
//==========================================================

template<typename scalar_t, typename F>
inline void hb_tile_elementwise_for(BSGTensor<scalar_t> res,
                               BSGTensor<scalar_t> input,
                               BSGTensor<scalar_t> other,
                               F functor) {
  char* data[3];
  data[0] = res.data_ptr();
  data[1] = input.data_ptr();
  data[2] = other.data_ptr();

  // is_trivial_1d
  if(res.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (res.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];

    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t len_per_tile = res.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
    size_t start = len_per_tile * __bsg_id;
    size_t end = start + len_per_tile;
    end = (end > res.numel())  ? res.numel() : end;
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + strides[0] * idx);
      scalar_t* input_dp = (scalar_t*)(data[1] + strides[1] * idx);
      scalar_t* other_dp = (scalar_t*)(data[2] + strides[2] * idx);
      *res_dp = functor(*input_dp, *other_dp);
    }
  } else {
    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t len_per_tile = res.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
    size_t start = len_per_tile * __bsg_id;
    size_t end = start + len_per_tile;
    end = (end > res.numel())  ? res.numel() : end;
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + offset_calc(idx, res));
      scalar_t* input_dp = (scalar_t*)(data[1] + offset_calc(idx, input));
      scalar_t* other_dp = (scalar_t*)(data[2] + offset_calc(idx, other));
      *res_dp = functor(*input_dp, *other_dp);
    }
  }
}

// =========================================================
// Tile Element-wise for -- Unary ops
//
// This function calculates the per tile range automatically
//==========================================================

template<typename scalar_t, typename F>
inline void hb_tile_elementwise_for(BSGTensor<scalar_t> res,
                               BSGTensor<scalar_t> input,
                               F functor) {
  char* data[2];
  data[0] = res.data_ptr();
  data[1] = input.data_ptr();

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
    size_t len_per_tile = res.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
    size_t start = len_per_tile * __bsg_id;
    size_t end = start + len_per_tile;
    end = (end > res.numel())  ? res.numel() : end;
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + strides[0] * idx);
      scalar_t* input_dp = (scalar_t*)(data[1] + strides[1] * idx);
      *res_dp = functor(*input_dp);
    }
  } else {
    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t len_per_tile = res.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
    size_t start = len_per_tile * __bsg_id;
    size_t end = start + len_per_tile;
    end = (end > res.numel())  ? res.numel() : end;
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + offset_calc(idx, res));
      scalar_t* input_dp = (scalar_t*)(data[1] + offset_calc(idx, input));
      *res_dp = functor(*input_dp);
    }
  }
}

// =========================================================
// Tile Element-wise for -- Nullary ops
//
// This function calculates the per tile range automatically
//==========================================================

template<typename scalar_t, typename F>
inline void hb_tile_elementwise_for(BSGTensor<scalar_t> res,
                               F functor) {
  char* data[1];
  data[0] = res.data_ptr();

  // is_trivial_1d
  if(res.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[1];
    strides[0] = (res.get_strides())[0];

    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t len_per_tile = res.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
    size_t start = len_per_tile * __bsg_id;
    size_t end = start + len_per_tile;
    end = (end > res.numel())  ? res.numel() : end;
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + strides[0] * idx);
      *res_dp = functor();
    }
  } else {
    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t len_per_tile = res.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
    size_t start = len_per_tile * __bsg_id;
    size_t end = start + len_per_tile;
    end = (end > res.numel())  ? res.numel() : end;
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + offset_calc(idx, res));
      *res_dp = functor();
    }
  }
}

// =========================================================
// BRG for
// =========================================================
// functor takes in current index

template <class FetchFunctor>
inline void brg_for(size_t numel, FetchFunctor functor) {
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
// BRG tile for
// =========================================================
// functor takes in current index

template <class FetchFunctor>
inline void brg_tile_for(size_t numel, FetchFunctor functor) {
  //--------------------------------------
  // calculate start and end for this tile
  //--------------------------------------
  size_t len_per_tile = numel / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > numel)  ? numel : end;
  //-----------------
  // loop
  //----------------
  for (size_t i = start; i < end; i++) {
    functor(i);
  }
}


#endif
