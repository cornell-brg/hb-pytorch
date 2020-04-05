#ifndef _HB_REDUCTION_H
#define _HB_REDUCTION_H

#include <hb_elementwise_for.hpp>

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

template<typename scalar_t, typename F1, typename F2>
inline void binary_reduction(BSGTensor<scalar_t>out,
                             BSGTensor<scalar_t>in,
                             uint32_t ndim, uint32_t num_reduction_dim,
                             uint32_t elements_per_output,
                             F1 reduce, F2 project) {
  switch(ndim) {
    case 1:
      // There is this corner case, in which each output is produced by only
      // one input element
      bsg_assert_msg(out.numel() == in.numel(),
                     "This case should be handled by reduction_simple?");
      brg_tile_for(out.numel(), [&](size_t n) {
        out(n) = project(in(n));
      });
      break;
    case 2:
      if(num_reduction_dim == 1) {
        // 2D input -- 1 reduction dim
        // parallelize over output elements
        brg_tile_for(out.numel(), [&](size_t n) {
          // reduction result init to 0
          scalar_t result = 0;
          for(size_t d = 0; d < elements_per_output; d++) {
            reduce(result, in(d, n));
          }
          out(0, n) = project(result);
        });
      } else {
        bsg_assert_msg(false, "Invalid number of reduction dims");
      }
      break;
    case 3:
      if(num_reduction_dim == 1) {
        // 3D input -- 1 reduction dim
        // parallelize over output elements
        brg_tile_for(out.numel(), [&](size_t n) {
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
        brg_tile_for(out.numel(), [&](size_t n) {
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
        bsg_assert_msg(false, "Invalid number of reduction dims");
      }
      break;
    default:
      bsg_assert_msg(false, "Invalid number of dims for reduction kernel");
  }
}

#endif
