//====================================================================
// index add kernel
// 10/22/2021 Niklas Schmelzle (jms854@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>
#include "bsg_manycore_atomic.h"

// equal to number of tiles
#define LOCK_SIZE 128


extern "C" {
  int lock[LOCK_SIZE] __attribute__ ((section (".dram")));

  void init_locks() {
    for (int linearIndex = bsg_id; linearIndex < LOCK_SIZE; linearIndex += BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM) {
        lock[linearIndex] = 0;
    }
  }

  void aquire_lock(int *lock) {
      int lock_ret = 1;
      do {
          lock_ret = bsg_amoswap_aq(lock, 1);
      } while (lock_ret != 0);
  }

  void release_lock(int *lock) {
      bsg_amoswap_rl(lock, 0);
  }


  int64_t get_element_index(HBTensor<float> &ten, int add_dim, int index, int64_t elementInSlice) {
      int64_t offset = 0;
      for (int i = ten.ndim()-1; i > 0; --i) {
          int32_t size = (i == add_dim)? 1 : ten.dim(i);
          offset += (elementInSlice % size) * ten.stride(i);
          elementInSlice /= size;
      }
      offset += elementInSlice * ten.stride(0) + index * ten.stride(add_dim);
      if (offset > ten.numel()) {
          bsg_printf("Warning: index out of range!\n");
          offset = 0;
      }
      return offset;
  }

  __attribute__ ((noinline))  int tensorlib_index_add_small_index(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          int64_t* dim_p,
          int64_t* sliceSize_p) {

    auto dst = HBTensor<float>(t0_p);
    auto src = HBTensor<float>(t1_p);
    auto idx = HBTensor<int64_t>(t2_p);
    int64_t dim = *dim_p;
    int64_t sliceSize = *sliceSize_p;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    init_locks();
    g_barrier.sync();

    for (int64_t srcIndex = 0; srcIndex < idx.numel(); ++srcIndex) {
        int64_t dstIndex = idx(srcIndex);
        for (int64_t linearIndex = bsg_id; linearIndex < sliceSize; linearIndex += BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM) {
            int64_t dst_element_idx = get_element_index(dst, dim, dstIndex, linearIndex);
            int64_t src_element_idx = get_element_index(src, dim, srcIndex, linearIndex);

            int64_t dst_lock_idx = dst_element_idx && 0xFF;
            int *dst_lock = &lock[dst_lock_idx];

            aquire_lock(dst_lock);
            dst(dst_element_idx) += src(src_element_idx);
            release_lock(dst_lock);
        }
    }

    //   End profiling
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }


  __attribute__ ((noinline))  int tensorlib_index_add_large_index(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          int64_t* dim_p,
          int64_t* sliceSize_p,
          int64_t* numIndices_p,
          int32_t* indexMajorMode_p) {

    auto dst = HBTensor<float>(t0_p);
    auto src = HBTensor<float>(t1_p);
    auto idx = HBTensor<int64_t>(t2_p);
    int64_t dim = *dim_p;
    int64_t sliceSize = *sliceSize_p;
    int64_t numIndices = *numIndices_p;
    int32_t indexMajorMode = *indexMajorMode_p;

      // Start profiling
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    init_locks();
    g_barrier.sync();


    for (int linearIndex = bsg_id; linearIndex < src.numel(); linearIndex += BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM) {
        int64_t srcIndex, elementInSlice;
        if (indexMajorMode == 1) {
            srcIndex = linearIndex / sliceSize;
            elementInSlice = linearIndex % sliceSize;
        } else {
            srcIndex = linearIndex % numIndices;
            elementInSlice = linearIndex / numIndices;
        }
        //bsg_printf("tile %d, srcIndex: %d, elementInSlice: %d\n", bsg_id, srcIndex, elementInSlice);
        int64_t dstIndex = idx(srcIndex);

        int64_t dst_element_idx = get_element_index(dst, dim, dstIndex, elementInSlice);
        int64_t src_element_idx = get_element_index(src, dim, srcIndex, elementInSlice);

        int64_t dst_lock_idx = dst_element_idx && 0xFF;
        int *dst_lock = &lock[dst_lock_idx];

        aquire_lock(dst_lock);
        dst(dst_element_idx) += src(src_element_idx);
        release_lock(dst_lock);
    }


    //   End profiling
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_index_add_small_index, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     int64_t*, int64_t*)
  HB_EMUL_REG_KERNEL(tensorlib_index_add_large_index, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     int64_t*, int64_t*, int64_t*, int32_t*)

}

