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
  Tensor int_vector = dense_vector.to(at::kInt);
  TORCH_CHECK(int_vector.dtype() == at::kInt, "Dense vector should be int !");
  TORCH_CHECK(self.dtype() == at::kInt, "C2SR should be int !");
  int64_t tensor_size = self.size(0);
  int32_t vector_size = (int32_t)int_vector.size(0);
//  std::cout << "tensor_size is " << tensor_size << std::endl;
  std::vector<int64_t> split_sizes(2);
  split_sizes[0] = tensor_size - 12;
  split_sizes[1] = 12;
//  std::cout << "split tensor in the first step" << std::endl;
//  std::cout << "split_sizes[0] is " << split_sizes[0] << std::endl;
//  std::cout << "split_sizes[1] is " << split_sizes[1] << std::endl;

  std::vector<Tensor> tensors = self.split_with_sizes(split_sizes, 0);
  Tensor other_hb = tensors[1];
  int64_t other_size = other_hb.size(0);
//  std::cout << "size of other_hb is " << other_size << std::endl;
  Tensor other_cpu = other_hb.to(at::DeviceType::CPU);
  int32_t *other_info = other_cpu.data_ptr<int32_t>();

  int32_t row = other_info[0];
  int32_t col = other_info[1];
  TORCH_CHECK(col == vector_size, "Number of columns in sparse matrix must be equal to vector size !");
  int32_t num_tile_x = other_info[2];
  int32_t num_tile_y = other_info[4];
  int32_t record_len = num_tile_x * num_tile_y;
//  for(int i = 0; i < 12; i++) {
//    std::cout << "other_info[" << i << "] is " << other_info[i] << std::endl;
//  }
  Tensor c2srandrecord = tensors[0];
  int64_t c2sr_size = c2srandrecord.size(0);
//  std::cout << "size of c2srandrecord is " << c2sr_size << std::endl;
 
  tensor_size = c2srandrecord.size(0);
  std::vector<int64_t> split_sizes1(2);
  split_sizes1[0] = tensor_size - record_len;
  split_sizes1[1] = record_len;
  std::vector<Tensor> tensors1 = c2srandrecord.split_with_sizes(split_sizes1, 0);
  
  Tensor c2sr_hb = tensors1[0];
  Tensor record_hb = tensors1[1];

  //Create a tensor that stores the other relative informations needed by the SpMV accelerators
/*  
  for (uint32_t tile_y_idx = 0; tile_y_idx < num_tile_y; tile_y_idx++) {
    int tile_y_dim = (tile_y_idx == num_tile_y - 1) ? last_tile_y : TILE_Y_SIZE;
    for (uint32_t tile_x_idx = 0; tile_x_idx < num_tile_x; tile_x_idx++) {
      int sparse_length[tile_y_dim];
      for(uint32_t i = tile_y_idx * TILE_Y_SIZE; i < tile_y_idx * TILE_Y_SIZE + tile_y_dim; i++) {
        uint32_t row_len = 0;
        int upper_bound = (tile_x_idx + 1) *TILE_X_SIZE;
        for(uint32_t j = tile_x_bound[i % TILE_Y_SIZE]; j < csr_ptr[i + 1]; j++) {
          if(csr_idx[j] < upper_bound) {
            row_len++;
            tile_x_bound[i % TILE_Y_SIZE]++;
          } else {
            break;
          }
        }
        sparse_length[i % TILE_Y_SIZE] = row_len;
      }
      uint32_t len_per_pe_a[NUM_PE];
      for(int i = 0; i < NUM_PE; i++) {
        len_per_pe_a[i] = 0;
      }
      for(int i = 0; i < tile_y_dim; i++) {
        len_per_pe_a[i % NUM_PE] += 2 + sparse_length[i] * 2;
      }
      uint32_t alloc_per_pe_a;
      uint32_t max_region_a = 0;
      for(int i=0; i<NUM_PE; i++){
        alloc_per_pe_a = ((len_per_pe_a[i] * 4) % (CACHELINE_BYTE) == 0) ?
                           ((len_per_pe_a[i] * 4) / CACHELINE_BYTE) : (((len_per_pe_a[i] * 4)/CACHELINE_BYTE) + 1);
        if (alloc_per_pe_a > max_region_a){
          max_region_a = alloc_per_pe_a;
        }
      }
      uint32_t length_total_a = max_region_a * CACHELINE_BYTE * NUM_PE;
      record_sparse_len[tile_y_idx * num_tile_x + tile_x_idx] = length_total_a;
      std::cout << "record_sparse_len[" << tile_y_idx * num_tile_x + tile_x_idx << "] is " << length_total_a << std::endl;
    }
  }
*/
  uint32_t cacheline_word = CACHELINE_BYTE / 4;
  uint32_t max_region_b = (((col + NUM_PE - 1) / NUM_PE) + cacheline_word - 1) / cacheline_word;
  uint32_t length_total_b = max_region_b * CACHELINE_BYTE * NUM_PE;
//  std::cout << "Convert the CPU data layout of dense vector to SpMV Xcel" << std::endl;
  Tensor vector_hb = at::empty({length_total_b / 4}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)}); 
  hb_offload_kernel(int_vector, vector_hb, "tensorlib_cputoxcel_vector");
//  std::cout << "Perform SpMV on Xcel" << std::endl;
  uint32_t max_region_c = (((row + NUM_PE - 1) / NUM_PE) + cacheline_word - 1) / cacheline_word;
  uint32_t length_total_c = max_region_c * CACHELINE_BYTE * NUM_PE;
//  std::cout << "Xcel_out is " << length_total_c << " and cpu out is " << row << std::endl;
  Tensor xcel_out = at::empty({length_total_c / 4}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  hb_offload_kernel(xcel_out, c2sr_hb, vector_hb, record_hb, other_hb, "tensorlib_spmvxcel");
//  std::cout << "Convert the SpMV Xcel data layout of dense vector to CPU" << std::endl;
  Tensor result = at::empty({row}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  hb_offload_kernel(result, xcel_out, "tensorlib_xceltocpu_vector");

  return result;
}

}} 
