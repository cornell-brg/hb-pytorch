//============================================================================
// The Tensor Indexing kernel
// 30/01/2021 Zhongyuan Zhao (zz546@cornell.edu)
//============================================================================


#include <kernel_common.hpp>

static int get(int idx, int num_indexers, int** indexer_ptr,  int* indexer_strides, int* original_sizes, int* original_strides) {
  int offset = 0;
//  printf("In get function: ");
  for(int j = 0; j < num_indexers; j++) {
//    printf("  original_strides[%d] is: %d\n", idx, original_strides[j]);
//    printf("  original_sizes[%d] is: %d\n", idx, original_sizes[j]);
//    printf("  indexer_strides[%d] is: %d\n", idx, indexer_strides[j]);
    int value = indexer_ptr[j][idx * indexer_strides[j]];
//    printf("  value %d is: %d\n", j, value);
    int size = original_sizes[j];
    if(value < 0) {
      value += size;
    }
    offset += value * original_strides[j] / sizeof(int);
  }
//  printf("  offset if idx %d is: %d\n", idx, offset);
  return offset;
}

static bool is_constant_index(int ntensors, int* strides) {
  hb_assert_msg(ntensors >= 3, "Number of tensors should be larger than 3\n");
  for (int arg = 2; arg < ntensors; arg++) {
    if (strides[arg] != 0) {
      return false;
    }
  }
  return true; 
}

static void calculate_values(int* values, int ndims, int* shapes) {
  int linear_offset = 0;
  for(int dim = 0; dim < ndims; dim++) {
    int size = shapes[dim];
    if(size > 0) {
      values[dim] = linear_offset % size;
//      printf("In calculate_values function, values[%d] is %d\n", dim, values[dim]);
      linear_offset /= size;
    }
  }
}

static void max_2d_step(int* step, int offset, int* shapes, int dim, int* values, int numel) {
  step[0] = std::min(shapes[0] - values[0], numel - offset);
//  printf("In max_2d_step function, out_sizes[0], values[0], numel and offset are %d, %d, %d, %d\n", shapes[0], values[0], numel, offset);
  step[1] = 1;
  if(step[0] == shapes[0] && dim >= 1) {
    step[1] = std::min(shapes[1] - values[1], (numel -offset) / shapes[0]);
  }
//  printf("In max_2d_step function, out_sizes[0], values[0] are %d, %d\n", shapes[1], values[1]);
}

static void increment(int* step, int& offset, int* shapes, int dim, int* values) {
  offset += step[0] * step[1];
  int overflow = step[0];
  int i = 0;
  if(step[1] != 1) {
    i = 1;
    overflow = step[1];
  }
  for(; i < dim && overflow > 0; i++) {
    int size = shapes[i];
    int prev = values[i];
    int value = prev + overflow;
    if(value >= size) {
      overflow = 1;
      value -= size;
    } else {
      overflow = 0;
    }
    values[i] = value;
//    printf("In increment function, value[%d] is %d\n", i, values[i]);
  }  
}

static void get_data_ptrs(int** ptrs, int* values, int* strides, int ndim, int ntensors) {
  for(int dim = 0; dim < ndim; dim++) {
    int value = values[dim];
    for(int arg = 0; arg < ntensors; arg++) {
      ptrs[arg] += value * strides[dim * ntensors + arg];
    }
  }
}

static void compute_core(int** ptrs, int* strides, int* index_sizes, int* index_strides, int ntensors, int step0, int step1) {
  size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
  int* outer_strides = &strides[ntensors];
  for(int i = 0; i < step1; i++) {
//    printf("In compute_core function, i is %d\n", i);
    if(i > 0) {
      for(int arg = 0; arg < ntensors; arg++) {
        ptrs[arg] += outer_strides[arg];
      }
    }   
    int* dst = ptrs[0];
    int* src = ptrs[1];
    if(is_constant_index(ntensors, strides)) {
//      printf("Before entering get function\n");
      int offset = get(0, ntensors-2, &ptrs[2], &strides[2], index_sizes, index_strides);
      for(int id = __bsg_id; id < step0; id = id + thread_num) {
//        printf("  dst offset -> strides[0] * id is: %d\n", strides[0] * id);
//        printf("  src offset - > strides[1] * id + offset is: %d\n", strides[1] * id + offset);
        *(dst + strides[0] * id) = *(src + strides[1] * id + offset);
      } 
    } else {
      for(int id = __bsg_id; id < step0; id = id + thread_num) {
        int offset = get(id, ntensors-2, &ptrs[2], &strides[2], index_sizes, index_strides);
//        printf("  dst offset -> strides[0] * id is: %d\n", strides[0] * id);
//        printf("  src offset - > strides[1] * id + offset is: %d\n", strides[1] * id + offset);
        *(dst + strides[0] * id) = *(src + strides[1] * id + offset);
      }
    }
  }
}

extern "C" {

__attribute__ ((noinline)) int tensorlib_indexing_1d(
  hb_tensor_t* _out,
  hb_tensor_t* _src,
  hb_tensor_t* _index1,
  hb_vector_t* _shapes,
  hb_vector_t* _kernel_strides,
  hb_vector_t* _index_sizes,
  hb_vector_t* _index_strides) {

  auto out_tensor = HBTensor<int>(_out);
  int* out = (int*)out_tensor.data_ptr();
  int* out_sizes = (int*)out_tensor.get_sizes();
  int out_numel = out_tensor.numel();
  int* src = (int*)HBTensor<int>(_src).data_ptr();
  int* index1 = (int*)HBTensor<int>(_index1).data_ptr();
  int* shapes = (int*)HBVector<int>(_shapes).data_ptr();
  int ndim = HBVector<int>(_shapes).numel();
  int* index_sizes = (int*)HBVector<int>(_index_sizes).data_ptr();
  int* index_strides = (int*)HBVector<int>(_index_strides).data_ptr();
  int* kernel_strides = (int*)HBVector<int>(_kernel_strides).data_ptr();

  int step[2];
  
  int* ptrs[3] = {out, src, index1};
  int values[ndim];
  calculate_values(values, ndim, out_sizes);
  bsg_cuda_print_stat_kernel_start();
  if(ndim <= 1) {
    get_data_ptrs(ptrs, values, kernel_strides, ndim, 3);
    compute_core(ptrs, kernel_strides, index_sizes, index_strides, 3, out_numel, 1);
  } else {
    int offset = 0;
    while(offset < out_numel) {
      get_data_ptrs(ptrs, values, kernel_strides, ndim, 3);
      max_2d_step(step, offset, shapes, ndim, values, out_numel);
      compute_core(ptrs, kernel_strides, index_sizes, index_strides, 3, step[0], step[1]);
      increment(step, offset, shapes, ndim, values);
    }
  }
  bsg_cuda_print_stat_kernel_end();
  g_barrier.sync();
  return 0;
}
  
HB_EMUL_REG_KERNEL(tensorlib_indexing_1d, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, 
                                          hb_vector_t*, hb_vector_t*, hb_vector_t*, hb_vector_t*)


__attribute__ ((noinline)) int tensorlib_indexing_2d(
  hb_tensor_t* _out,
  hb_tensor_t* _src,
  hb_tensor_t* _index1,
  hb_tensor_t* _index2,
  hb_vector_t* _shapes,
  hb_vector_t* _kernel_strides,
  hb_vector_t* _index_sizes,
  hb_vector_t* _index_strides) {

  auto out_tensor = HBTensor<int>(_out);
  int* out = (int*)out_tensor.data_ptr();
  int* out_sizes = (int*)out_tensor.get_sizes();
  int out_numel = out_tensor.numel();
  int* src = (int*)HBTensor<int>(_src).data_ptr();
  int* index1 = (int*)HBTensor<int>(_index1).data_ptr();
  int* index2 = (int*)HBTensor<int>(_index2).data_ptr();
  int* shapes = (int*)HBVector<int>(_shapes).data_ptr();
  int ndim = HBVector<int>(_shapes).numel();
  int* index_sizes = (int*)HBVector<int>(_index_sizes).data_ptr();
  int* index_strides = (int*)HBVector<int>(_index_strides).data_ptr();
  int* kernel_strides = (int*)HBVector<int>(_kernel_strides).data_ptr();

  int step[2];

  int* ptrs[4] = {out, src, index1, index2};
  int values[ndim];
  calculate_values(values, ndim, out_sizes);

  bsg_cuda_print_stat_kernel_start();
  if(ndim <= 1) {
    get_data_ptrs(ptrs, values, kernel_strides, ndim, 4);
    compute_core(ptrs, kernel_strides, index_sizes, index_strides, 4, out_numel, 1);
  } else {
    int offset = 0;
    while(offset < out_numel) {
      get_data_ptrs(ptrs, values, kernel_strides, ndim, 4);
      max_2d_step(step, offset, shapes, ndim, values, out_numel);
      compute_core(ptrs, kernel_strides, index_sizes, index_strides, 4, step[0], step[1]);
      increment(step, offset, shapes, ndim, values);
    }
  }
  bsg_cuda_print_stat_kernel_end();
  g_barrier.sync(); 
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib_indexing_2d, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, 
                                          hb_vector_t*, hb_vector_t*, hb_vector_t*, hb_vector_t*)
    
}
