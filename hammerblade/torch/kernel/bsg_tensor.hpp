#ifndef _BSG_TENSOR_HPP
#define _BSG_TENSOR_HPP

#include <math.h>
#include <initializer_list>

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
  uint32_t strides;
  uint32_t data;
} bsg_tensor_t;

typedef struct {
  uint32_t N;
  uint32_t data;
} bsg_vector_t;

// =========================================================
// Device Tensor classes
//
// Wrapper classes around device tensor structs to provide
// convenience operations.
// =========================================================

class BSGTensor {
  private:
    uint32_t N;
    uint32_t dims;
    uint32_t* strides;
    float* data;

  public:
    BSGTensor(bsg_tensor_t* t) :
      N(t->N),
      dims(t->dims),
      strides((uint32_t*) ((intptr_t) t->strides)),
      data((float*) ((intptr_t) t->data)) {}

    int size() {
      return N;
    }

    int dim() {
      return dims;
    }

    template<typename... T>
    float operator()(T... indices) {
      std::initializer_list<uint32_t> iarray = {indices...};

      if(iarray.size() != dims) {
        bsg_printf("BSGTensor error: number of indices must be %d, given %d\n",
            dims, iarray.size());
        return NAN;
      }

      uint32_t offset = 0;
      uint32_t s = 0;
      for(auto index : iarray) {
        offset += (index * strides[s]);
        s++;
      }

      if(offset >= N) {
        bsg_printf("BSGTensor error: index out of bounds\n");
        return NAN;
      }

      return data[offset];
    }
};

#endif // _BSG_TENSOR_HPP
