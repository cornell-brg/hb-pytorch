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
    idx = (uint32_t)((float)idx / (float)sizes[i]);
    offset += (uint32_t)((float)dimx * (float)strides[i]);
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
  size_t pod_size     = pod_end - pod_start;
  // per tile range within a pod
  size_t len_per_tile = pod_size / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start        = len_per_tile * __bsg_id;
  size_t end          = start + len_per_tile;
  end = (end > pod_size) ? pod_size : end;
  // range in global idx
  range->start = pod_start + start;
  range->end   = pod_start + end;

  return;
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

    //------------------------------
    // in the case where stride is 0
    //------------------------------
    scalar_t fixed_data[4];
    for (size_t i = 0; i < 4; i++) {
      if (strides[i] == 0) {
        fixed_data[i] = *(scalar_t*)data[i];
        data[i] = (char*)&fixed_data[i];
      }
    }

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

    //------------------------------
    // in the case where stride is 0
    //------------------------------
    scalar_t fixed_data[3];
    for (size_t i = 0; i < 3; i++) {
      if (strides[i] == 0) {
        fixed_data[i] = *(scalar_t*)data[i];
        data[i] = (char*)&fixed_data[i];
      }
    }

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

    //------------------------------
    // in the case where stride is 0
    //------------------------------
    scalar_t fixed_data[2];
    for (size_t i = 0; i < 2; i++) {
      if (strides[i] == 0) {
        fixed_data[i] = *(scalar_t*)data[i];
        data[i] = (char*)&fixed_data[i];
      }
    }

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

    //------------------------------
    // in the case where stride is 0
    //------------------------------
    scalar_t fixed_data[1];
    for (size_t i = 0; i < 1; i++) {
      if (strides[i] == 0) {
        fixed_data[i] = *(scalar_t*)data[i];
        data[i] = (char*)&fixed_data[i];
      }
    }

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

    //------------------------------
    // in the case where stride is 0
    //------------------------------
    scalar_t fixed_data[4];
    for (size_t i = 0; i < 4; i++) {
      if (strides[i] == 0) {
        fixed_data[i] = *(scalar_t*)data[i];
        data[i] = (char*)&fixed_data[i];
      }
    }

    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    hb_range range;
    calc_range(&range, res.numel());
    size_t start = range.start;
    size_t end   = range.end;

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
    hb_range range;
    calc_range(&range, res.numel());
    size_t start = range.start;
    size_t end   = range.end;

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

    //------------------------------
    // in the case where stride is 0
    //------------------------------
    scalar_t fixed_data[3];
    for (size_t i = 0; i < 3; i++) {
      if (strides[i] == 0) {
        fixed_data[i] = *(scalar_t*)data[i];
        data[i] = (char*)&fixed_data[i];
      }
    }

    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    hb_range range;
    calc_range(&range, res.numel());
    size_t start = range.start;
    size_t end   = range.end;

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
    hb_range range;
    calc_range(&range, res.numel());
    size_t start = range.start;
    size_t end   = range.end;

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

    //------------------------------
    // in the case where stride is 0
    //------------------------------
    scalar_t fixed_data[2];
    for (size_t i = 0; i < 2; i++) {
      if (strides[i] == 0) {
        fixed_data[i] = *(scalar_t*)data[i];
        data[i] = (char*)&fixed_data[i];
      }
    }

    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    hb_range range;
    calc_range(&range, res.numel());
    size_t start = range.start;
    size_t end   = range.end;

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
    hb_range range;
    calc_range(&range, res.numel());
    size_t start = range.start;
    size_t end   = range.end;

    for (size_t idx = start; idx < end; idx++) {
      scalar_t* res_dp = (scalar_t*)(data[0] + offset_calc(idx, res));
      scalar_t* input_dp = (scalar_t*)(data[1] + offset_calc(idx, input));
      *res_dp = functor(*input_dp);
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
    hb_range range;
    calc_range(&range, res.numel());
    size_t start = range.start;
    size_t end   = range.end;

    data[0] += strides[0] * start;
    data[1] += strides[1] * start;
    for (size_t idx = start; idx < end; idx++) {
      scalar_dst* res_dp = (scalar_dst*)(data[0]);
      scalar_src* input_dp = (scalar_src*)(data[1]);
      *res_dp = functor(*input_dp);
      data[0] += strides[0];
      data[1] += strides[1];
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
      scalar_dst* res_dp = (scalar_dst*)(data[0] + offset_calc(idx, res));
      scalar_src* input_dp = (scalar_src*)(data[1] + offset_calc(idx, input));
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

    //------------------------------
    // in the case where stride is 0
    //------------------------------
    scalar_t fixed_data[1];
    for (size_t i = 0; i < 1; i++) {
      if (strides[i] == 0) {
        fixed_data[i] = *(scalar_t*)data[i];
        data[i] = (char*)&fixed_data[i];
      }
    }

    //-----------------------------
    // iterating over all elementes
    //-----------------------------
    hb_range range;
    calc_range(&range, res.numel());
    size_t start = range.start;
    size_t end   = range.end;

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
    hb_range range;
    calc_range(&range, res.numel());
    size_t start = range.start;
    size_t end   = range.end;

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


#endif
