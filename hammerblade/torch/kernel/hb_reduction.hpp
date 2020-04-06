#ifndef _HB_REDUCTION_H
#define _HB_REDUCTION_H

#include <hb_parallel_for.hpp>

//====================================================================
// Reduction mode used in LossNLL and other loss functions
//====================================================================

enum Reduction {
  None,             // Do not reduce
  Mean,             // (Possibly weighted) mean of losses
  Sum,              // Sum losses
  END
};

//====================================================================
// Binary reductions -- sum, mean, etc.
//
// ndim -> number of dims in reduction iterator -- this may not equal
//         to the input shape
// num_reduction_dim -> number of dims to be reduced
// elements_per_output -> how many input elements to create one output
// reduce -> how to process a new input element
// project -> how to do postprocessing on result
//
// 04/02/2020 Bandhav Veluri and Lin Cheng
//====================================================================

// Potential cases:
// 1D input -- 1 reduction dim -- trivial
// 2D input -- 1 reduction dim
// 2D input -- 2 reduction dim -- trivial
// 3D input -- 1 reduction dim
// 3D input -- 2 reduction dim
// 3D input -- 3 reduction dim -- trivial
// 4D input -- 1 reduction dim
// 4D input -- 2 reduction dim
// 4D input -- 3 reduction dim
// 4D input -- 4 reduction dim -- trivial

// Trivial case -- reduce to 1 output

template<typename scalar_t, typename F1, typename F2>
inline void binary_reduction_simple(BSGTensor<scalar_t> out,
                                    BSGTensor<scalar_t> in,
                                    F1 reduce, F2 project) {
  bsg_assert_msg(out.numel() == 1, "reduction_simple only handles trivial case");

  if(__bsg_id == 0) {

    char* data[2];
    data[0] = out.data_ptr();
    data[1] = in.data_ptr();

    //-----------------------------
    // partial_result
    //-----------------------------
    scalar_t result = 0;

    // is_trivial_1d
    if(in.ndim() == 1) {


      //-----------------------------
      // collect metadata
      //-----------------------------
      uint32_t strides[2];
      strides[0] = (out.get_strides())[0];
      strides[1] = (in.get_strides())[0];

      //-----------------------------
      // iterating over all elementes
      //-----------------------------
      size_t start = 0;
      size_t end = in.numel();
      for (size_t idx = start; idx < end; idx++) {
        // XXX: when offloading through reduction path, strides are measured in numel
        scalar_t* in_dp = (scalar_t*)(data[1] + strides[1] * idx * sizeof(scalar_t));
        reduce(result, *in_dp);
      }
    } else {
      //-----------------------------
      // iterating over all elementes
      //-----------------------------
      size_t start = 0;
      size_t end = in.numel();
      for (size_t idx = start; idx < end; idx++) {
        // XXX: when offloading through reduction path, strides are measured in numel
        scalar_t* in_dp = (scalar_t*)(data[1] + offset_calc(idx, in) * sizeof(scalar_t));
        reduce(result, *in_dp);
      }
    }

    // produce final result
    scalar_t* out_dp = (scalar_t*)(data[0]);
    *out_dp = project(result);
  }
}

template<typename scalar_t, typename F1, typename F2>
inline void binary_reduction(HBTensor<scalar_t>out,
                             HBTensor<scalar_t>in,
                             uint32_t ndim, uint32_t num_reduction_dim,
                             uint32_t elements_per_output,
                             F1 reduce, F2 project) {
  if(out.numel() == 1) {
    binary_reduction_simple(out, in, reduce, project);
    return;
  }

  switch(ndim) {
    case 1:
      // There is this corner case, in which each output is produced by only
      // one input element
      hb_assert_msg(out.numel() == in.numel(),
                     "This case should be handled by reduction_simple?");
      hb_parallel_for(out.numel(), [&](size_t n) {
        out(n) = project(in(n));
      });
      break;
    case 2:
      if(num_reduction_dim == 1) {
        // 2D input -- 1 reduction dim
        // parallelize over output elements
        hb_parallel_for(out.numel(), [&](size_t n) {
          // reduction result init to 0
          scalar_t result = 0;
          for(size_t d = 0; d < elements_per_output; d++) {
            reduce(result, in(d, n));
          }
          out(0, n) = project(result);
        });
      } else {
        hb_assert_msg(false, "Invalid number of reduction dims");
      }
      break;
    case 3:
      if(num_reduction_dim == 1) {
        // 3D input -- 1 reduction dim
        // parallelize over output elements
        hb_parallel_for(out.numel(), [&](size_t n) {
          // reduction result init to 0
          scalar_t result = 0;
          uint32_t dim1 = n / in.dim(2);
          uint32_t dim2 = n % in.dim(2);
          for(size_t d = 0; d < elements_per_output; d++) {
            reduce(result, in(d, dim1, dim2));
          }
          out(0, dim1, dim2) = project(result);
        });
      } else if(num_reduction_dim == 2) {
        // 3D input -- 2 reduction dim
        // parallelize over output elements
        hb_parallel_for(out.numel(), [&](size_t n) {
          // reduction result init to 0
          scalar_t result = 0;
          for(size_t d = 0; d < elements_per_output; d++) {
            uint32_t dim0 = d / in.dim(1);
            uint32_t dim1 = d % in.dim(1);
            reduce(result, in(dim0, dim1, n));
          }
          out(0, 0, n) = project(result);
        });
      } else {
        hb_assert_msg(false, "Invalid number of reduction dims");
      }
      break;
    default:
      hb_assert_msg(false, "Invalid number of dims for reduction kernel");
  }
}

#endif
