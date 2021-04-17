//====================================================================
//The Pytorch C++ layer for xcelspmv operator
//10/30/2020 Zhongyuan Zhao (zz546@cornell.edu)
//=====================================================================
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/hammerblade/SparseCommon.hpp>
#include <chrono>

namespace at { namespace native {

Tensor xcelspmv_hb(const Tensor &self, const Tensor &dense_vector) {
  TORCH_CHECK(self.dim() == 1, "1D matrix expected, got dim ", self.dim(), " tensor");
  Tensor int_vector;
  if (!(dense_vector.dtype() == at::kInt)) {
    int_vector = dense_vector.to(at::kInt);
  } else {
    int_vector = dense_vector;
  }
  TORCH_CHECK(int_vector.dtype() == at::kInt, "Dense vector should be int !");
  TORCH_CHECK(self.dtype() == at::kInt, "C2SR should be int !");
  int64_t tensor_size = self.size(0);
  int32_t vector_size = (int32_t)int_vector.size(0);

  std::vector<int64_t> split_sizes(2);
  split_sizes[0] = tensor_size - 12;
  split_sizes[1] = 12;

  std::vector<Tensor> tensors = self.split_with_sizes(split_sizes, 0);
  Tensor other_hb = tensors[1];
  int64_t other_size = other_hb.size(0);

  Tensor other_cpu = other_hb.to(at::DeviceType::CPU);
  int32_t *other_info = other_cpu.data_ptr<int32_t>();

  int32_t row = other_info[0];
  int32_t col = other_info[1];
  TORCH_CHECK(col == vector_size, "Number of columns in sparse matrix must be equal to vector size !");
  int32_t num_tile_x = other_info[2];
  int32_t num_tile_y = other_info[4];
  int32_t record_len = num_tile_x * num_tile_y;

  Tensor c2srandrecord = tensors[0];
  int64_t c2sr_size = c2srandrecord.size(0);
 
  tensor_size = c2srandrecord.size(0);
  std::vector<int64_t> split_sizes1(2);
  split_sizes1[0] = tensor_size - record_len;
  split_sizes1[1] = record_len;
  std::vector<Tensor> tensors1 = c2srandrecord.split_with_sizes(split_sizes1, 0);
  
  Tensor c2sr_hb = tensors1[0];
  Tensor record_hb = tensors1[1];

//  uint32_t cacheline_word = CACHELINE_BYTE / 4;
//  uint32_t max_region_b = (((col + NUM_PE - 1) / NUM_PE) + cacheline_word - 1) / cacheline_word;
//  uint32_t length_total_b = max_region_b * CACHELINE_BYTE * NUM_PE;
//  std::cout << "Convert the CPU data layout of dense vector to SpMV Xcel" << std::endl;
//  Tensor vector_hb = at::empty({length_total_b / 4}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)}); 
//  hb_offload_kernel(int_vector, vector_hb, "tensorlib_cputoxcel_vector");
//  std::cout << "Perform SpMV on Xcel" << std::endl;
//  uint32_t max_region_c = (((row + NUM_PE - 1) / NUM_PE) + cacheline_word - 1) / cacheline_word;
//  uint32_t length_total_c = max_region_c * CACHELINE_BYTE * NUM_PE;
//  std::cout << "Xcel_out is " << length_total_c << " and cpu out is " << row << std::endl;
//  Tensor xcel_out = at::empty({length_total_c / 4}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  Tensor result = at::empty({row}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  hb_offload_kernel(result, c2sr_hb, int_vector, record_hb, other_hb, "tensorlib_spmvxcel");
//  std::cout << "Convert the SpMV Xcel data layout of dense vector to CPU" << std::endl;
//  hb_offload_kernel(result, xcel_out, "tensorlib_xceltocpu_vector");

  return result;
}

}} 
