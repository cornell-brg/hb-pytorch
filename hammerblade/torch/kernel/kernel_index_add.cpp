//====================================================================
// index add kernel
// 10/22/2021 Niklas Schmelzle (jms854@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>
#include "bsg_manycore_arch.h"
#include "bsg_mcs_mutex.hpp"


#define MTX_SIZE 256


extern "C" {
  bsg_mcs_mutex_t mtx[MTX_SIZE] __attribute__ ((section (".dram")));


  int get_element_index(HBTensor<float> &ten, int add_dim, int index, int elementInSlice) {
      int offset = 0;
      for (int i = ten.ndim()-1; i > 0; --i) {
          int size = (i == add_dim)? 1 : ten.dim(i);
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
          int32_t* dim_p,
          int32_t* sliceSize_p) {

    auto dst = HBTensor<float>(t0_p);
    auto src = HBTensor<float>(t1_p);
    auto idx = HBTensor<int32_t>(t2_p);
    int dim = *dim_p;
    int sliceSize = *sliceSize_p;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    bsg_mcs_mutex_node_t lcl, *lcl_as_glbl = (bsg_mcs_mutex_node_t*)bsg_tile_group_remote_ptr(int, bsg_x, bsg_y, &lcl);
    g_barrier.sync();

    for (int srcIndex = 0; srcIndex < idx.numel(); ++srcIndex) {
        int dstIndex = idx(srcIndex);
        for (int linearIndex = bsg_id; linearIndex < sliceSize; linearIndex += BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM) {
            int dst_element_idx = get_element_index(dst, dim, dstIndex, linearIndex);
            int src_element_idx = get_element_index(src, dim, srcIndex, linearIndex);

            int dst_mtx_idx = dst_element_idx && 0xFF;
            bsg_mcs_mutex_t *dst_mtx = &mtx[dst_mtx_idx];

            bsg_mcs_mutex_acquire(dst_mtx, &lcl, lcl_as_glbl);
            dst(dst_element_idx) += src(src_element_idx);
            bsg_mcs_mutex_release(dst_mtx, &lcl, lcl_as_glbl);
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
          int32_t* dim_p,
          int32_t* sliceSize_p,
          int32_t* numIndices_p,
          int32_t* indexMajorMode_p) {

    auto dst = HBTensor<float>(t0_p);
    auto src = HBTensor<float>(t1_p);
    auto idx = HBTensor<int32_t>(t2_p);
    int dim = *dim_p;
    int sliceSize = *sliceSize_p;
    int numIndices = *numIndices_p;
    int indexMajorMode = *indexMajorMode_p;

      // Start profiling
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    bsg_mcs_mutex_node_t lcl, *lcl_as_glbl = (bsg_mcs_mutex_node_t*)bsg_tile_group_remote_ptr(int, bsg_x, bsg_y, &lcl);
    g_barrier.sync();


    for (int linearIndex = bsg_id; linearIndex < src.numel(); linearIndex += BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM) {
        int srcIndex, elementInSlice;
        if (indexMajorMode == 1) {
            srcIndex = linearIndex / sliceSize;
            elementInSlice = linearIndex % sliceSize;
        } else {
            srcIndex = linearIndex % numIndices;
            elementInSlice = linearIndex / numIndices;
        }
        //bsg_printf("tile %d, srcIndex: %d, elementInSlice: %d\n", bsg_id, srcIndex, elementInSlice);
        int dstIndex = idx(srcIndex);

        int dst_element_idx = get_element_index(dst, dim, dstIndex, elementInSlice);
        int src_element_idx = get_element_index(src, dim, srcIndex, elementInSlice);

        int dst_mtx_idx = dst_element_idx && 0xFF;
        bsg_mcs_mutex_t *dst_mtx = &mtx[dst_mtx_idx];

        bsg_mcs_mutex_acquire(dst_mtx, &lcl, lcl_as_glbl);
        dst(dst_element_idx) += src(src_element_idx);
        bsg_mcs_mutex_release(dst_mtx, &lcl, lcl_as_glbl);
    }


    //   End profiling
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_index_add_small_index, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     int32_t*, int32_t*)
  HB_EMUL_REG_KERNEL(tensorlib_index_add_large_index, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     int32_t*, int32_t*, int32_t*, int32_t*)

}

