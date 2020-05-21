//====================================================================
// Tensor Data Structures
// 03/09/2020 Bandhav Veluri and Lin Cheng (lc873@cornell.edu)
//====================================================================

#ifndef _HB_TENSOR_HPP
#define _HB_TENSOR_HPP

#include <math.h>
#include <initializer_list>
#include <cstdint>
#include <hb_assert.hpp>
#include <hb_hw_patch.hpp>

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

// Info about storage objects
#ifdef HB_ENABLE_KERNEL_LOG
  void* storage_head;
  uint32_t storage_numel;
#endif
} hb_tensor_t;

typedef struct {
  uint32_t N;
#ifdef HB_EMUL
  uint64_t data;
#else
  uint32_t data;
#endif
} hb_vector_t;

// =========================================================
// Device Tensor classes
//
// Wrapper classes around device tensor structs to provide
// convenience operations. This runs on a tiny RISC-V processor
// on the device, so be careful about using dynamic memory
// allocation.
// =========================================================

template <typename DT, uint32_t dims=-1>
class HBTensor {
  private:
    uint32_t N;
    uint32_t strides[dims];
    uint32_t sizes[dims];
    DT* data;

  public:
    HBTensor(hb_tensor_t* t) :
      N(t->N),
      data((DT*) ((intptr_t) t->data)) {
        // WAW HW bug seems to be triggered on a non-bloacking load to
        // the register holding `sizes` in various kernels. This fix
        // adds a RAW dependedncy on that register, blocking the load.
        HB_FIX_WAW_HAZARD(sizes);

        hb_assert_msg(
          t->dims == dims,
          "error: HBTensor dims don't match offloaed tensor dims");

        uint32_t* strides_remote = (uint32_t*) ((intptr_t) t->strides);
        uint32_t* sizes_remote = (uint32_t*) ((intptr_t) t->sizes);

        // Move strides and sizes to DRAM
        for(int i=0; i<dims; ++i) {
          strides[i] = strides_remote[i];
          sizes[i] = sizes_remote[i];
        }
      }

    char* data_ptr() {
      return (char*)data;
    }

    uint32_t* get_strides() {
      return strides;
    }

    uint32_t* get_sizes() {
      return sizes;
    }

    int numel() {
      return N;
    }

    uint32_t dim(uint32_t d) {
      hb_assert_msg(d < dims,
                    "error: dimesnion must be less than %d\n",
                    dims);
      return sizes[d];
    }

    uint32_t ndim() {
      return dims;
    }

    // Special case where we want linear, 0-d
    // and 1-d tensor indexing.
    //
    // XXX: The tensor has to be contiguous if
    // it's >1-d tensor.
    DT& operator()(uint32_t index) {
      hb_assert_msg(index < N,
                    "error: N=%d but accessed %d\n",
                    N, index);
      if(dims != 1) {
        return data[index];
      } else {
        // Explicitly calculate data index to handle
        // non-contiguous 1-d tensors.
        return data[index * strides[0]];
      }
    }

    template<typename... T>
    DT& operator()(uint32_t index0, T... indices) {
      std::initializer_list<uint32_t> iarray = {index0, indices...};

      hb_assert_msg(iarray.size() == dims,
                    "error: expected dims=%d arguments but got %d\n",
                    dims, iarray.size());
      uint32_t offset = 0;
      uint32_t s = 0;
      for(auto index : iarray) {
        offset += (index * strides[s]);
        s++;
      }

      hb_assert_msg(offset < N,
                    "error: N=%d but accessed %d\n",
                    N, offset);

      return data[offset];
    }
};

template <typename DT>
class HBTensor<DT, -1> {
  private:
    uint32_t N;
    uint32_t dims;
    uint32_t* strides;
    uint32_t* sizes;
    DT* data;

  public:
    HBTensor(hb_tensor_t* t) :
      N(t->N),
      dims(t->dims),
      strides((uint32_t*) ((intptr_t) t->strides)),
      sizes((uint32_t*) ((intptr_t) t->sizes)),
      data((DT*) ((intptr_t) t->data)) {
        // WAW HW bug seems to be triggered on a non-bloacking load to
        // the register holding `sizes` in various kernels. This fix
        // adds a RAW dependedncy on that register, blocking the load.
        HB_FIX_WAW_HAZARD(sizes);
      }

    char* data_ptr() {
      return (char*)data;
    }

    uint32_t* get_strides() {
      return strides;
    }

    uint32_t* get_sizes() {
      return sizes;
    }

    int numel() {
      return N;
    }

    uint32_t dim(uint32_t d) {
      hb_assert_msg(d < dims,
                    "error: dimesnion must be less than %d\n",
                    dims);
      return sizes[d];
    }

    uint32_t ndim() {
      return dims;
    }

    // Special case where we want linear, 0-d
    // and 1-d tensor indexing.
    //
    // XXX: The tensor has to be contiguous if
    // it's >1-d tensor.
    DT& operator()(uint32_t index) {
      hb_assert_msg(index < N,
                    "error: N=%d but accessed %d\n",
                    N, index);
      if(dims != 1) {
        return data[index];
      } else {
        // Explicitly calculate data index to handle
        // non-contiguous 1-d tensors.
        return data[index * strides[0]];
      }
    }

    template<typename... T>
    DT& operator()(uint32_t index0, T... indices) {
      std::initializer_list<uint32_t> iarray = {index0, indices...};

      hb_assert_msg(iarray.size() == dims,
                    "error: expected dims=%d arguments but got %d\n",
                    dims, iarray.size());
      uint32_t offset = 0;
      uint32_t s = 0;
      for(auto index : iarray) {
        offset += (index * strides[s]);
        s++;
      }

      hb_assert_msg(offset < N,
                    "error: N=%d but accessed %d\n",
                    N, offset);

      return data[offset];
    }
};

template<typename T>
class HBVector {
  private:
    uint32_t N;
    T* data;

  public:
    HBVector(hb_vector_t* v) :
      N(v->N), data((T*) ((intptr_t) v->data)) {}

    uint32_t numel() {
      return N;
    }

    T& operator[](uint32_t i) {
      return data[i];
    }
};

#endif // _HB_TENSOR_HPP
