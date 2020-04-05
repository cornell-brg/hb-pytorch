//====================================================================
// Tensor Data Structures
// 03/09/2020 Bandhav Veluri and Lin Cheng (lc873@cornell.edu)
//====================================================================

#ifndef _BSG_TENSOR_HPP
#define _BSG_TENSOR_HPP

#include <math.h>
#include <initializer_list>
#include <bsg_assert.hpp>

// =========================================================
// Device Tensor structs
//
// These structs are used by the device to move tensors (and
// as a special case, vectors) back and forth the host. As a
// these structs have to maintain the exact same memory layout
// to those on the host. A consequence of that is these have
// to be pure C struct.
// =========================================================

typedef struct {
  uint32_t N;
  uint32_t dims;
#ifdef HB_EMUL
  uint64_t strides;
  uint64_t sizes;
  uint64_t data;
#else
  uint32_t strides;
  uint32_t sizes;
  uint32_t data;
#endif
} bsg_tensor_t;

typedef struct {
  uint32_t N;
#ifdef HB_EMUL
  uint64_t data;
#else
  uint32_t data;
#endif
} bsg_vector_t;

// =========================================================
// Device Tensor classes
//
// Wrapper classes around device tensor structs to provide
// convenience operations. This runs on a tiny RISC-V processor
// on the device, so be careful about using dynamic memory
// allocation.
// =========================================================

template <typename DT>
class BSGTensor {
  private:
    uint32_t N;
    uint32_t dims;
    uint32_t* strides;
    uint32_t* sizes;
    DT* data;

  public:
    BSGTensor(bsg_tensor_t* t) :
      N(t->N),
      dims(t->dims),
      strides((uint32_t*) ((intptr_t) t->strides)),
      sizes((uint32_t*) ((intptr_t) t->sizes)),
      data((DT*) ((intptr_t) t->data)) {}

    int numel() {
      return N;
    }

    uint32_t dim(uint32_t d) {
      bsg_assert_msg(d < dims,
                     "error: dimesnion must be less than %d\n",
                     dims);
      return sizes[d];
    }

    uint32_t ndim() {
      return dims;
    }

    template<typename T>
    DT& operator[](std::initializer_list<T> iarray) {
      //std::initializer_list<uint32_t> iarray = {indices...};

      // special case where we have a 0-dim tensor
      if(dims == 0) {
        bsg_assert_msg(iarray.size() == 1,
                       "error: expected only one argument with 0-dim tensor\n");
        for(auto index : iarray) {
          bsg_assert_msg(index == 0,
                         "error: index must be 0 0-dim tensor\n");
        }
        return data[0];
      }

      // special case where we want linear indexing
      // when dims != 1
      // XXX: this tensor has to be contiguous
      if(iarray.size() == 1 && dims != 1) {
        for(auto index : iarray) {
          return data[index];
        }
      }

      bsg_assert_msg(iarray.size() == dims,
                     "error: expected dims=%d arguments but got %d\n",
                     dims, iarray.size());
      uint32_t offset = 0;
      uint32_t s = 0;
      for(auto index : iarray) {
        offset += (index * strides[s]);
        s++;
      }

      bsg_assert_msg(offset < N, "error: N=%d but accessed %d\n", N, offset);

      return data[offset];
    }
};

template<typename T>
class BSGVector {
  private:
    uint32_t N;
    T* data;

  public:
    BSGVector(bsg_vector_t* v) :
      N(v->N), data((T*) ((intptr_t) v->data)) {}

    uint32_t numel() {
      return N;
    }

    T& operator[](uint32_t i) {
      return data[i];
    }
};

#endif // _BSG_TENSOR_HPP
