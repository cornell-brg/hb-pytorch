//====================================================================
// Element-wise for helper function
// 03/12/2020 Lin Cheng (lc873@cornell.edu)
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
  for(int32_t i = tensor.ndim() - 1; i >= 0; i--) {
    uint32_t dimx = idx % sizes[i];
    idx /= sizes[i];
    offset += dimx * strides[i];
  }
  return offset;
}

// =========================================================
// Pointwise for -- Ternary
// =========================================================

template<typename scalar_t, typename F>
inline void hb_foreach(HBTensor<scalar_t> res,
                       HBTensor<scalar_t> input,
                       HBTensor<scalar_t> tensor1,
                       HBTensor<scalar_t> tensor2,
                       F functor) {
  char* data[4];
  data[0] = res.data_ptr();
  data[1] = input.data_ptr();
  data[2] = tensor1.data_ptr();
  data[3] = tensor2.data_ptr();

  // is_trivial_1d
  if(res.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[4];
    strides[0] = (res.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (tensor1.get_strides())[0];
    strides[3] = (tensor2.get_strides())[0];

    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t start = 0;
    size_t end = res.numel();
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0]);
      scalar_t* input_dp = (scalar_t*)(data[1]);
      scalar_t* tensor1_dp = (scalar_t*)(data[2]);
      scalar_t* tensor2_dp = (scalar_t*)(data[3]);
      *res_dp = functor(*input_dp, *tensor1_dp, *tensor2_dp);
      data[0] += strides[0];
      data[1] += strides[1];
      data[2] += strides[2];
      data[3] += strides[3];
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
      scalar_t* tensor1_dp = (scalar_t*)(data[2] + offset_calc(idx, tensor1));
      scalar_t* tensor2_dp = (scalar_t*)(data[3] + offset_calc(idx, tensor2));
      *res_dp = functor(*input_dp, *tensor1_dp, *tensor2_dp);
    }
  }
}

// =========================================================
// Elementwise for -- Binary ops
// =========================================================

template<typename scalar_t, typename F>
inline void hb_foreach(HBTensor<scalar_t> res,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
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
      scalar_t* res_dp = (scalar_t*)(data[0]);
      scalar_t* input_dp = (scalar_t*)(data[1]);
      scalar_t* other_dp = (scalar_t*)(data[2]);
      *res_dp = functor(*input_dp, *other_dp);
      data[0] += strides[0];
      data[1] += strides[1];
      data[2] += strides[2];
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
inline void hb_foreach(HBTensor<scalar_t> res,
                               HBTensor<scalar_t> input,
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
      scalar_t* res_dp = (scalar_t*)(data[0]);
      scalar_t* input_dp = (scalar_t*)(data[1]);
      *res_dp = functor(*input_dp);
      data[0] += strides[0];
      data[1] += strides[1];
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
inline void hb_foreach(HBTensor<scalar_t> res,
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
      scalar_t* res_dp = (scalar_t*)(data[0]);
      *res_dp = functor();
      data[0] += strides[0];
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
// Tile Pointwise for -- Ternary
// =========================================================

template<typename scalar_t, typename F>
inline void hb_tiled_foreach(HBTensor<scalar_t> res,
                                HBTensor<scalar_t> input,
                                HBTensor<scalar_t> tensor1,
                                HBTensor<scalar_t> tensor2,
                                F functor) {
  char* data[4];
  data[0] = res.data_ptr();
  data[1] = input.data_ptr();
  data[2] = tensor1.data_ptr();
  data[3] = tensor2.data_ptr();

  // is_trivial_1d
  if(res.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[4];
    strides[0] = (res.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (tensor1.get_strides())[0];
    strides[3] = (tensor2.get_strides())[0];

    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    size_t len_per_tile = res.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
    size_t start = len_per_tile * __bsg_id;
    size_t end = start + len_per_tile;
    end = (end > res.numel())  ? res.numel() : end;
    data[0] += strides[0] * start;
    data[1] += strides[1] * start;
    data[2] += strides[2] * start;
    data[3] += strides[3] * start;
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0]);
      scalar_t* input_dp = (scalar_t*)(data[1]);
      scalar_t* tensor1_dp = (scalar_t*)(data[2]);
      scalar_t* tensor2_dp = (scalar_t*)(data[3]);
      *res_dp = functor(*input_dp, *tensor1_dp, *tensor2_dp);
      data[0] += strides[0];
      data[1] += strides[1];
      data[2] += strides[2];
      data[3] += strides[3];
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
      scalar_t* tensor1_dp = (scalar_t*)(data[2] + offset_calc(idx, tensor1));
      scalar_t* tensor2_dp = (scalar_t*)(data[3] + offset_calc(idx, tensor2));
      *res_dp = functor(*input_dp, *tensor1_dp, *tensor2_dp);
    }
  }
}

// =========================================================
// Tile Element-wise for -- Binary ops
//
// This function calculates the per tile range automatically
//==========================================================

template<typename scalar_t, typename F>
inline void hb_tiled_foreach(HBTensor<scalar_t> res,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
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
    data[0] += strides[0] * start;
    data[1] += strides[1] * start;
    data[2] += strides[2] * start;
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0]);
      scalar_t* input_dp = (scalar_t*)(data[1]);
      scalar_t* other_dp = (scalar_t*)(data[2]);
      *res_dp = functor(*input_dp, *other_dp);
      data[0] += strides[0];
      data[1] += strides[1];
      data[2] += strides[2];
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
inline void hb_tiled_foreach(HBTensor<scalar_t> res,
                               HBTensor<scalar_t> input,
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
    data[0] += strides[0] * start;
    data[1] += strides[1] * start;
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0]);
      scalar_t* input_dp = (scalar_t*)(data[1]);
      *res_dp = functor(*input_dp);
      data[0] += strides[0];
      data[1] += strides[1];
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
inline void hb_tiled_foreach(HBTensor<scalar_t> res,
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
    data[0] += strides[0] * start;
    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0]);
      *res_dp = functor();
      data[0] += strides[0];
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
