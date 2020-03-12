//====================================================================
// Element-wise for helper function
// 03/12/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#ifndef _BRG_ELEMENT_FOR_HPP
#define _BRG_ELEMENT_FOR_HPP

#include <cassert>
#include <math.h>
#include <bsg_tensor.hpp>
#include <initializer_list>

// =========================================================
// Element-wise for
// We would like the synatx to be like this:
// __attribute__ ((noinline))  int tensorlib_add(bsg_tensor_t* c_p,
//                                               bsg_tensor_t* a_p,
//                                               bsg_tensor_t* b_p,
//                                               float* alpha) {
//   brg_tile_element_wise_for<float*>(c_p, a_p, b_p, [&]() {
//      *c = *a + alpha * (*b);
//   });
// }

// =========================================================
// Device Tensor that comes from TensorIterator, in which its
// strides are measured in **bytes**
//
// Helper class to make our lives easier when writing element
// -wise operation kernels
// Note: here strides could be *0*
//
// You may call ++ to advance the iterator, but be noted that
// it is monolithic. There is no way to return
// =========================================================

template <typename T>
class BRGIteratorTensor {
  private:
    uint32_t N;
    uint32_t dims;
    uint32_t strides;
    uint32_t data;
    uint32_t cur_loc;

  public:
    BRGIteratorTensor(bsg_tensor_t* t, uint32_t start_loc = 0) :
      N(t->N),
      dims(t->dims),
      strides(*(uint32_t*)((intptr_t) t->strides)),
      data((intptr_t) t->data),
      cur_loc(start_loc) {
        assert(cur_loc < N);
        assert(dims == 1);
      }

    BRGIteratorTensor& operator ++ (int) {
      data += strides;
      cur_loc++;
      assert(cur_loc < N);
      return *this;
    }

    T operator*() {
      return (T)(intptr_t)data;
    }

};

#endif
