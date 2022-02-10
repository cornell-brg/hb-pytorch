//====================================================================
// Scatter add kernel
// Feb/07/2022 Zhongyuan Zhao (zz546@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <hb_reduction.hpp>

#ifdef HB_EMUL
#include <mutex>
#else
#include "bsg_manycore_arch.h"
#include "bsg_mcs_mutex.hpp"
#endif

#define MTX_SIZE 256

template<typename scalar_t>
inline uint32_t cal_offset(HBTensor<scalar_t> tensor, uint32_t* vals) {
  uint32_t* strides = tensor.get_strides();
  uint32_t offset = 0;
  for (uint32_t i = 0; i < tensor.ndim(); i++) {
    offset += vals[i] * strides[i];
  }
  return offset;
}

template<typename scalar_t>
inline void cal_cods(HBTensor<scalar_t> tensor, uint32_t* vals) {
  uint32_t* strides = tensor.get_strides();
  uint32_t* sizes = tensor.get_sizes();

  for(uint32_t i = 0; i < tensor.ndim(); i++) {
    vals[i] = vals[i] + 1;
    if (vals[i] != sizes[i])
      break;
    else
      vals[i] = 0;
  }
}

extern "C" {

  #ifdef HB_EMUL
  std::mutex matrix[MTX_SIZE];
  #else
  bsg_mcs_mutex_t matrix[MTX_SIZE] __attribute__ ((section (".dram")));
  #endif  
  
  __attribute__ ((noinline))  int tensorlib_scatter_add(
          hb_tensor_t* _self,
          hb_tensor_t* _index,
          hb_tensor_t* _src,
          uint32_t* _dim) {
    auto self = HBTensor<float>(_self);
    auto index = HBTensor<int32_t>(_index);
    auto src = HBTensor<float>(_src);

    int   *index_data = (int*)index.data_ptr();
    float *self_data  = (float*)self.data_ptr();
    float *src_data  = (float*)src.data_ptr();

    uint32_t dim = *_dim;
    auto ndim = index.ndim();
    uint32_t num = index.numel();
    uint32_t num_threads = bsg_tiles_X * bsg_tiles_Y;

    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();
    #ifndef HB_EMUL
    bsg_mcs_mutex_node_t lcl, *lcl_as_glbl = (bsg_mcs_mutex_node_t*)bsg_tile_group_remote_ptr(int, bsg_x, bsg_y, &lcl);
    g_barrier.sync();
    #endif

    int num_elements = num % num_threads == 0 ? (num / num_threads) : (num / num_threads + 1);
    int start = __bsg_id * num_elements;
    int end = (start + num_elements) < num ? (start + num_elements) : num;
    uint32_t cods[ndim];
    uint32_t self_cods[ndim];
    uint32_t* sizes = index.get_sizes();

    int init_idx = start;
    for(uint32_t i = 0; i < ndim; i++) {
      uint32_t dimx = init_idx % sizes[i];
      cods[i] = dimx;
      init_idx /= sizes[i];
    }

    for(uint32_t i = start; i < end; i++) {
      if(i != start) {
        cal_cods(index, cods);
      }
      uint32_t idx_offset = cal_offset(index, cods);
      for(int j = 0; j < ndim; j++) {
        self_cods[j] = cods[j];
      }
      
      self_cods[dim] = *(index_data + idx_offset);
      uint32_t self_offset = cal_offset(self, self_cods);
      uint32_t src_offset = cal_offset(src, cods);
      uint32_t dst_matrix_idx = self_offset && 0xFF;
      #ifdef HB_EMUL
      std::mutex *dst_matrix = &matrix[dst_matrix_idx];
      dst_matrix->lock();
      #else
      bsg_mcs_mutex_t *dst_matrix = &matrix[dst_matrix_idx];
      bsg_mcs_mutex_acquire(dst_matrix, &lcl, lcl_as_glbl);
      #endif
      *(self_data + self_offset) += *(src_data + src_offset);
      #ifdef HB_EMUL
      dst_matrix->unlock();
      #else
      bsg_mcs_mutex_release(dst_matrix, &lcl, lcl_as_glbl);
      #endif
    }   
      
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_scatter_add, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, uint32_t*)

}
