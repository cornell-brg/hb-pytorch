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

    uint32_t numel() {
      return N;
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

template <typename T, typename F>
inline void brg_element_wise_for(bsg_tensor_t* _t0, bsg_tensor_t* _t1,
                                 bsg_tensor_t* _t2, F functor) {
  auto res = BRGIteratorTensor<T*>(_t0);
  auto input = BRGIteratorTensor<T*>(_t1);
  auto other = BRGIteratorTensor<T*>(_t2);
  size_t start = 0;
  size_t end = res.numel();
  for (size_t i = start; i < end; i++) {
    *(*res) = functor(*(*input), *(*other));
    res++;
    input++;
    other++;
  }
}

#endif
