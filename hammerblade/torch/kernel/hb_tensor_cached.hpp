//====================================================================
// Cached Tensor
// 03/09/2020 Bandhav Veluri
//====================================================================

#ifndef _HB_TENSOR_CACHED_HPP
#define _HB_TENSOR_CACHED_HPP

#include <cstdint>
#include <bsg_manycore.h>
#include <hb_assert.hpp>
#include <hb_common.hpp>
#include <hb_tensor.hpp>

//typedef struct {
//  uint32_t tag;
//  float data;
//} hb_tensor_cache_line_t;

template <typename DT, int32_t dims=-1, uint32_t cache_size = 8>
class HBTensorCached : public HBTensorImpl<__remote DT, uint32_t> {
  private:
    uint32_t strides[dims];
    uint32_t sizes[dims];
    const uint32_t cache_numel = cache_size / sizeof(hb_tensor_cache_line_t);
    hb_tensor_cache_line_t cache[cache_size / sizeof(hb_tensor_cache_line_t)] = {0};

  public:
    HBTensorCached(hb_tensor_t* t) :
      HBTensorImpl<__remote DT, uint32_t>(
        t->N,
        (uint32_t) dims,
        strides,
        sizes,
        (__remote DT*) ((intptr_t) t->data)
      ) {
        hb_assert_msg(
          t->dims == dims,
          "error: HBTensor dims don't match offloaed tensor dims");

        uint32_t* strides_remote = (uint32_t*) ((intptr_t) t->strides);
        uint32_t* sizes_remote = (uint32_t*) ((intptr_t) t->sizes);

        // Move strides and sizes to scratchpad
        for(int i=0; i<dims; ++i) {
          strides[i] = strides_remote[i];
          sizes[i] = sizes_remote[i];
        }
      }
    
    template<typename ...T>
    DT cached_read(T... indices) {
      uint32_t offset = offset(indices...);
      uint32_t ci = offset % cache_numel;

      if(cache[ci].tag == offset) {
        return cache[ci].data;
      }

      float rdata = data[offset];
      cache[ci].tag = offset;
      cache[ci].data = rdata;
      return rdata;
    }
};

#endif // _HB_TENSOR_CACHED_HPP
