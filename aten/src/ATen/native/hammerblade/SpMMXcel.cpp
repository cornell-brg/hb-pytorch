//====================================================================
//The Pytorch C++ layer for mm_xcel operator
//20/02/2021 Zhongyuan Zhao (zz546@cornell.edu)
//====================================================================
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/hammerblade/SparseCommon.hpp>
#include <chrono>

namespace at { namespace native {

Tensor xcelspmm_hb(const Tensor &self, const Tensor &dense_matrix) {
  TORCH_CHECK(self.dim() == 1, "1D matrix expected, got dim ", self.dim(), " tensor");
  Tensor int_matrix;
  if(!(dense_matrix.dtype() == at::kInt)) {
    int_matrix = dense_matrix.to(at::kInt);
  } else {
    int_matrix = dense_matrix;
  }
  TORCH_CHECK(int_matrix.dtype() == at::kInt, "Dense vector should be int !");
  int64_t tensor_size = self.size(0);
  std::vector<int64_t> split_sizes(2);
  split_sizes[0] = tensor_size - 5;
  split_sizes[1] = 5;
  std::vector<Tensor> tensors = self.split_with_sizes(split_sizes, 0);

  int32_t n = (int32_t)dense_matrix.size(0);
  int32_t k = (int32_t)dense_matrix.size(1);
  
  Tensor c2sr_hb = tensors[0];
  Tensor other_hb = tensors[1];
  Tensor other_cpu = other_hb.to(at::DeviceType::CPU);
  int32_t *other_info = other_cpu.data_ptr<int32_t>();

  int32_t m = other_info[0];  

  uint32_t cacheline_word = CACHELINE_BYTE / 4;
  uint32_t max_region_b = ( ((n + NUM_PE - 1)/NUM_PE) * k + cacheline_word - 1) / cacheline_word;
  uint32_t length_total_b = max_region_b * CACHELINE_BYTE * NUM_PE;
  
  Tensor matrix_hb = at::empty({length_total_b / 4}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  hb_offload_kernel(int_matrix, matrix_hb, "tensorlib_cputoxcel_matrix"); 
  uint32_t length_total_c = ((((m + NUM_PE - 1)/NUM_PE)* k + cacheline_word - 1) / cacheline_word) * CACHELINE_BYTE * NUM_PE; 
  Tensor xcel_out = at::empty({length_total_c / 4}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  hb_offload_kernel(xcel_out, c2sr_hb, matrix_hb, other_hb, n, k, "tensorlib_spmmxcel");
  Tensor result = at::empty({m, k}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  hb_offload_kernel(result, xcel_out, "tensorlib_xceltocpu_matrix");  
  return result; 
}

}}
